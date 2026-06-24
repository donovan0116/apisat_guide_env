"""
Phase 2: GAT-MAPPO Training for SemGAT-MARL.

Algorithm 1 (lines 6-10) of the paper:
    1. Load the frozen VAE encoder from Phase 1 pretraining
    2. Replace standard MLP actor/critic with GAT-based networks
    3. At each step, build the semantic interaction graph
    4. Train with MAPPO (centralized GAT critic, decentralized GAT actor)
    5. Apply semantic-aware reward shaping (paper Eq. 4)

Usage:
    # With pretrained VAE
    python scripts/semgat_train.py --vae_checkpoint models/semgat/vae_pretrained.pt \
        --num_drones 4 --num_targets 4 --total_steps 500000

    # Without VAE (use raw obs directly in GAT — ablation baseline)
    python scripts/semgat_train.py --no_vae --num_drones 4 --num_targets 4
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.gat_policy import GATActor, GATCritic
from core.semantic import (
    HeuristicSemanticClassifier,
    SemanticGraph,
    make_semantic_classifier,
)
from core.vae import DualHeadVAE
from envs import ParallelQuadrotorDelivery
from utils.config import FullConfig


# ── Constants (same as mappo_train.py) ──────────────────────────────────

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
TANH_EPS = 1e-6


# ── Running normalization ───────────────────────────────────────────────


class RunningMeanStd:
    """Running mean and standard deviation for online normalization."""

    def __init__(self, shape: tuple, eps: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = eps

    def update(self, x: np.ndarray):
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


# ── Rollout buffer with graph support ───────────────────────────────────


class SemGATRolloutBuffer:
    """Rollout buffer that stores per-step graph structure for GAT updates."""

    def __init__(
        self,
        n_steps: int,
        n_agents: int,
        obs_dim: int,
        global_dim: int,
        act_dim: int,
        latent_dim: int,
        max_edges: int = 200,
    ):
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.global_dim = global_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.max_edges = max_edges
        self.reset()

    def reset(self):
        self.ptr = 0
        self.obs = np.zeros((self.n_steps, self.n_agents, self.obs_dim), dtype=np.float32)
        self.global_obs = np.zeros((self.n_steps, self.global_dim), dtype=np.float32)
        self.latent_z = np.zeros((self.n_steps, self.n_agents, self.latent_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_agents, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)
        self.values = np.zeros(self.n_steps, dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_agents), dtype=np.bool_)
        self.masks = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)

        # Graph data per step (padded to max_edges)
        self.edge_index = np.zeros((self.n_steps, 2, self.max_edges), dtype=np.int64)
        self.edge_type = np.zeros((self.n_steps, self.max_edges), dtype=np.int64)
        self.n_edges = np.zeros(self.n_steps, dtype=np.int64)  # actual num edges per step

    def add(
        self,
        obs: np.ndarray,
        global_obs: np.ndarray,
        latent_z: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        value: float,
        dones: np.ndarray,
        masks: np.ndarray,
        edge_index: np.ndarray,
        edge_type: np.ndarray,
    ):
        if self.ptr >= self.n_steps:
            raise RuntimeError("SemGATRolloutBuffer overflow")
        self.obs[self.ptr] = obs
        self.global_obs[self.ptr] = global_obs
        self.latent_z[self.ptr] = latent_z
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = value
        self.dones[self.ptr] = dones
        self.masks[self.ptr] = masks

        # Store graph data (truncate/pad to max_edges)
        n_e = min(edge_index.shape[1], self.max_edges)
        self.n_edges[self.ptr] = n_e
        self.edge_index[self.ptr, :, :n_e] = edge_index[:, :n_e]
        self.edge_type[self.ptr, :n_e] = edge_type[:n_e]

        self.ptr += 1

    def data(self) -> Dict[str, np.ndarray]:
        return {
            "obs": self.obs[: self.ptr],
            "global_obs": self.global_obs[: self.ptr],
            "latent_z": self.latent_z[: self.ptr],
            "actions": self.actions[: self.ptr],
            "log_probs": self.log_probs[: self.ptr],
            "rewards": self.rewards[: self.ptr],
            "values": self.values[: self.ptr],
            "dones": self.dones[: self.ptr],
            "masks": self.masks[: self.ptr],
            "edge_index": self.edge_index[: self.ptr],
            "edge_type": self.edge_type[: self.ptr],
            "n_edges": self.n_edges[: self.ptr],
        }


# ── Trainer ─────────────────────────────────────────────────────────────


class SemGATMAPPOTrainer:
    """SemGAT-MARL trainer: VAE features → GAT policy → MAPPO optimization."""

    def __init__(self, config: FullConfig):
        self.config = config
        self.device = torch.device(config.train.device)
        torch.manual_seed(config.train.seed)
        np.random.seed(config.train.seed)

        # Probe environment
        probe_env = self._make_env()
        obs, info = probe_env.reset(seed=config.train.seed)
        agent0 = probe_env.possible_agents[0]
        self.agent_ids = probe_env.possible_agents
        self.n_agents = len(self.agent_ids)
        self.obs_dim = probe_env.observation_space(agent0).shape[0]
        self.act_dim = probe_env.action_space(agent0).shape[0]
        self.global_dim = info["global_obs"].shape[0]
        probe_env.close()

        train_cfg = config.train
        semgat_cfg = config.semgat

        # ── VAE (frozen feature extractor) ──
        self.vae: Optional[DualHeadVAE] = None
        self.latent_dim = semgat_cfg.gat_hidden_dim  # fallback if no VAE
        if semgat_cfg.use_vae_latent and semgat_cfg.vae_checkpoint:
            self.vae = self._load_vae(semgat_cfg.vae_checkpoint)
            self.latent_dim = self.vae.latent_dim
            print(f"[SemGAT] Loaded VAE from {semgat_cfg.vae_checkpoint}")
        else:
            print("[SemGAT] No VAE loaded — using raw observations as GAT input")

        # ── Semantic classifier ──
        self.sem_classifier = make_semantic_classifier(
            use_llm=config.vae.use_llm,
            llm_model=config.vae.llm_model,
            llm_api_key=config.vae.llm_api_key,
        )

        # ── GAT Actor ──
        self.actor = GATActor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            latent_dim=self.latent_dim,
            use_latent=semgat_cfg.use_vae_latent,
            gat_hidden=semgat_cfg.gat_hidden_dim,
            gat_heads=semgat_cfg.gat_n_heads,
            gat_layers=semgat_cfg.gat_n_layers,
            num_edge_types=5,
            dropout=semgat_cfg.gat_dropout,
        ).to(self.device)

        # ── GAT Critic ──
        self.critic = GATCritic(
            global_dim=self.global_dim,
            latent_dim=self.latent_dim,
            use_latent=semgat_cfg.use_vae_latent,
            gat_hidden=semgat_cfg.gat_hidden_dim,
            gat_heads=semgat_cfg.gat_n_heads,
            gat_layers=semgat_cfg.gat_n_layers,
            num_edge_types=5,
            dropout=semgat_cfg.gat_dropout,
        ).to(self.device)

        # ── Optimizers ──
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=train_cfg.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=train_cfg.learning_rate
        )

        # ── Buffer ──
        self.buffer = SemGATRolloutBuffer(
            train_cfg.n_steps,
            self.n_agents,
            self.obs_dim,
            self.global_dim,
            self.act_dim,
            self.latent_dim,
            max_edges=200,
        )

        # ── Normalization ──
        self.obs_rms = RunningMeanStd((self.obs_dim,))
        self.global_obs_rms = RunningMeanStd((self.global_dim,))

    def _load_vae(self, checkpoint_path: str) -> DualHeadVAE:
        """Load a pretrained DH-VAE and freeze its encoder."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        vae_cfg = checkpoint["config"]["vae"]
        vae = DualHeadVAE(
            obs_dim=checkpoint["obs_dim"],
            latent_dim=checkpoint["latent_dim"],
            max_neighbors=checkpoint["max_neighbors"],
            num_classes=5,
            encoder_hidden=vae_cfg.get("encoder_hidden", [256, 128]),
            decoder_hidden=vae_cfg.get("decoder_hidden", [128, 256]),
        ).to(self.device)
        vae.load_state_dict(checkpoint["vae_state_dict"])
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        return vae

    def _make_env(self) -> ParallelQuadrotorDelivery:
        cfg = self.config
        return ParallelQuadrotorDelivery(
            num_drones=cfg.env.num_drones,
            num_targets=cfg.env.num_targets,
            num_obstacles=cfg.env.num_obstacles,
            task_mode=cfg.env.task_mode,
            bounds=cfg.env.bounds_array,
            quad_params=cfg.quad,
            obs_config=cfg.obs,
            act_config=cfg.act,
            reward_config=cfg.reward,
            term_config=cfg.term,
            aggregate_phy_steps=cfg.env.aggregate_phy_steps,
            seed=cfg.train.seed,
        )

    def _normalize_obs(self, obs: np.ndarray, update: bool = True, mask: np.ndarray = None) -> np.ndarray:
        flat = obs.reshape(-1, self.obs_dim)
        if update:
            if mask is not None:
                active_mask = mask.reshape(-1) > 0.5
                if active_mask.any():
                    self.obs_rms.update(flat[active_mask])
            else:
                self.obs_rms.update(flat)
        return self.obs_rms.normalize(flat).reshape(obs.shape)

    def _normalize_global_obs(self, global_obs: np.ndarray, update: bool = True) -> np.ndarray:
        g = global_obs[None, :] if global_obs.ndim == 1 else global_obs
        if update:
            self.global_obs_rms.update(g)
        return self.global_obs_rms.normalize(g).reshape(global_obs.shape)

    @torch.no_grad()
    def _extract_latent(self, obs_norm: np.ndarray) -> np.ndarray:
        """Extract VAE latent features from normalized observations."""
        if self.vae is None:
            return np.zeros((obs_norm.shape[0], self.latent_dim), dtype=np.float32)
        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=self.device)
        z = self.vae.extract_features(obs_t)
        return z.cpu().numpy()

    def _get_env_state(self, base_env):
        """Extract state arrays from the underlying QuadrotorDeliveryEnv."""
        target_positions = np.array([t.position for t in base_env._targets])
        return {
            "states": base_env._states.copy(),
            "target_positions": target_positions,
            "target_assignment": base_env._target_assignment.copy(),
            "obstacle_positions": base_env._obstacle_positions.copy(),
            "obstacle_radii": base_env._obstacle_radii.copy(),
            "carry_status": (
                base_env._carry_status.copy()
                if base_env._carry_status is not None
                else None
            ),
        }

    def _build_graph(self, base_env) -> Tuple[np.ndarray, np.ndarray]:
        """Build semantic interaction graph from the underlying env state.

        Returns (edge_index, edge_type) filtered to agent-to-agent edges
        plus self-loops, suitable as input to the GAT policy.
        """
        sd = self._get_env_state(base_env)
        graph = self.sem_classifier.classify(
            sd["states"],
            sd["target_positions"],
            sd["target_assignment"],
            sd["obstacle_positions"],
            sd["obstacle_radii"],
            carry_status=sd["carry_status"],
        )
        n_agents = graph.n_agents

        # Filter: keep only agent-to-agent edges (Separate relations)
        agent_edges = [e for e in graph.edges if e.src < n_agents and e.dst < n_agents]

        # Build edge_index and edge_type, including self-loops
        edge_list = []
        type_list = []
        for i in range(n_agents):
            edge_list.append((i, i))
            type_list.append(0)  # No-Relation for self
        for e in agent_edges:
            edge_list.append((e.src, e.dst))
            type_list.append(e.relation)

        if not edge_list:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_type = np.zeros((0,), dtype=np.int64)
        else:
            edge_index = np.array(edge_list, dtype=np.int64).T  # (2, E)
            edge_type = np.array(type_list, dtype=np.int64)  # (E,)

        return edge_index, edge_type

    # ── Semantic reward shaping (paper Eq. 4) ──────────────────────────

    def _semantic_reward(
        self,
        cur_state_data: dict,
        prev_state_data: dict,
        base_rewards: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Apply semantic-aware reward shaping.

        Adds:
            +w_T for approaching a Target relation
            +w_S for maintaining safe distance from Avoid/Separate entities
        """
        if not self.config.semgat.use_semantic_reward:
            return base_rewards

        cfg = self.config.semgat
        states = cur_state_data["states"]
        prev_states = prev_state_data.get("states", states)
        target_positions = cur_state_data["target_positions"]
        target_assignment = cur_state_data["target_assignment"]

        shaped = base_rewards.copy()

        for i in range(self.n_agents):
            if masks[i] < 0.5:
                continue
            aid = target_assignment[i]
            if aid >= 0 and aid < len(target_positions):
                prev_dist = float(np.linalg.norm(prev_states[i, :3] - target_positions[aid]))
                curr_dist = float(np.linalg.norm(states[i, :3] - target_positions[aid]))
                # Reward distance reduction toward Target
                shaped[i] += cfg.rew_target_approach * (prev_dist - curr_dist)

            # Reward for safe inter-agent distance
            pos_i = states[i, :3]
            for j in range(self.n_agents):
                if i == j or masks[j] < 0.5:
                    continue
                dist = float(np.linalg.norm(pos_i - states[j, :3]))
                safe_dist = 5.0  # safety radius
                if dist > safe_dist:
                    shaped[i] += cfg.rew_safe_distance * 0.01  # small bonus for safe separation

        return shaped

    # ── Training loop ──────────────────────────────────────────────────

    def train(self):
        cfg = self.config.train
        env = self._make_env()
        base_env = env.env  # underlying QuadrotorDeliveryEnv for state access
        os.makedirs(cfg.model_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        writer = SummaryWriter(cfg.log_dir) if SummaryWriter else None

        obs_dict, info = env.reset(seed=cfg.seed)
        global_obs = info["global_obs"].astype(np.float32)
        total_steps = 0
        updates = 0
        episode_idx = 0
        episode_reward = 0.0
        episode_len = 0
        reward_window = deque(maxlen=100)
        start_time = time.time()

        print(
            f"[SemGAT] training {cfg.total_timesteps} env steps | "
            f"{self.n_agents} drones | obs={self.obs_dim} act={self.act_dim} "
            f"latent={self.latent_dim}"
        )

        # Track previous state for semantic reward shaping
        prev_state_data = self._get_env_state(base_env)

        while total_steps < cfg.total_timesteps:
            self.buffer.reset()

            for _ in range(cfg.n_steps):
                active_agents = set(env.agents)
                obs_batch = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
                actions_batch = np.zeros((self.n_agents, self.act_dim), dtype=np.float32)
                log_probs = np.zeros(self.n_agents, dtype=np.float32)
                masks = np.zeros(self.n_agents, dtype=np.float32)

                for i, agent_id in enumerate(self.agent_ids):
                    if agent_id not in active_agents:
                        continue
                    obs_batch[i] = obs_dict[agent_id]
                    masks[i] = 1.0

                # Normalize observations
                norm_obs_batch = self._normalize_obs(obs_batch, mask=masks)
                norm_global_obs = self._normalize_global_obs(global_obs)

                # Extract VAE latent features
                latent_z_batch = self._extract_latent(norm_obs_batch)

                # Build semantic graph from underlying env state
                edge_index, edge_type = self._build_graph(base_env)

                # Forward pass through GAT
                obs_t = torch.as_tensor(norm_obs_batch, dtype=torch.float32, device=self.device)
                latent_t = torch.as_tensor(latent_z_batch, dtype=torch.float32, device=self.device)
                global_t = torch.as_tensor(norm_global_obs, dtype=torch.float32, device=self.device)
                ei_t = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
                et_t = torch.as_tensor(edge_type, dtype=torch.long, device=self.device)

                with torch.no_grad():
                    action_tensor, log_prob_tensor, _ = self.actor.sample(
                        obs_t, latent_t, ei_t, et_t
                    )
                    value = self.critic(global_t, latent_t, ei_t, et_t).item()

                actions_batch[:] = action_tensor.cpu().numpy()
                log_probs[:] = log_prob_tensor.cpu().numpy()
                actions = {
                    agent_id: actions_batch[i]
                    for i, agent_id in enumerate(self.agent_ids)
                }

                next_obs, rewards, terms, truncs, next_info = env.step(actions)
                reward_batch = np.array(
                    [rewards.get(agent_id, 0.0) for agent_id in self.agent_ids],
                    dtype=np.float32,
                )
                done_batch = np.array(
                    [
                        terms.get(agent_id, False) or truncs.get(agent_id, False)
                        for agent_id in self.agent_ids
                    ],
                    dtype=np.bool_,
                )

                # Semantic reward shaping using current vs. previous state
                cur_state_data = self._get_env_state(base_env)
                shaped_rewards = self._semantic_reward(
                    cur_state_data, prev_state_data, reward_batch, masks
                )
                prev_state_data = cur_state_data

                self.buffer.add(
                    norm_obs_batch,
                    norm_global_obs,
                    latent_z_batch,
                    actions_batch,
                    log_probs,
                    shaped_rewards,
                    value,
                    done_batch,
                    masks,
                    edge_index,
                    edge_type,
                )

                total_steps += 1
                episode_len += 1
                episode_reward += float((shaped_rewards * masks).sum())

                obs_dict = next_obs
                global_obs = next_info["global_obs"].astype(np.float32)

                # Episode end
                if not env.agents:
                    reward_window.append(episode_reward)
                    episode_idx += 1
                    if episode_idx % cfg.n_eval_episodes == 0:
                        avg = np.mean(reward_window) if reward_window else 0.0
                        fps = int(total_steps / max(time.time() - start_time, 1e-6))
                        if writer:
                            writer.add_scalar("rollout/episode_return", episode_reward, total_steps)
                            writer.add_scalar("rollout/episode_return_mean_100", avg, total_steps)
                            writer.add_scalar("rollout/episode_length", episode_len, total_steps)
                            writer.add_scalar("time/fps", fps, total_steps)
                        print(
                            f"step={total_steps:>8d} update={updates:>5d} "
                            f"episode={episode_idx:>5d} avg_return={avg:>9.2f} "
                            f"last_len={episode_len:>4d} fps={fps:>4d}"
                        )
                    obs_dict, info = env.reset()
                    global_obs = info["global_obs"].astype(np.float32)
                    prev_state_data = self._get_env_state(base_env)
                    episode_reward = 0.0
                    episode_len = 0

                if total_steps >= cfg.total_timesteps:
                    break

            # ── PPO Update ──
            if self.buffer.ptr > 1:
                next_value = self._bootstrap_value(global_obs, base_env, env.agents)
                update_info = self._update(self.buffer.data(), next_value)
                updates += 1

                # Learning rate annealing
                progress = min(1.0, total_steps / cfg.total_timesteps)
                lr = cfg.learning_rate * (1.0 - progress)
                for pg in self.actor_optimizer.param_groups:
                    pg["lr"] = lr
                for pg in self.critic_optimizer.param_groups:
                    pg["lr"] = lr

                if writer and update_info:
                    writer.add_scalar("train/actor_loss", update_info["policy_loss"], total_steps)
                    writer.add_scalar("train/critic_loss", update_info["value_loss"], total_steps)
                    writer.add_scalar("train/entropy", update_info["entropy"], total_steps)
                    writer.add_scalar("train/learning_rate", lr, total_steps)

            # ── Evaluation ──
            if updates > 0 and (total_steps % cfg.eval_freq < cfg.n_steps):
                eval_info = self.evaluate(num_episodes=cfg.n_eval_episodes)
                if writer:
                    writer.add_scalar("eval/avg_return", eval_info["avg_return"], total_steps)
                    writer.add_scalar("eval/avg_length", eval_info["avg_length"], total_steps)
                    writer.add_scalar(
                        "eval/avg_targets_reached", eval_info["avg_targets_reached"], total_steps
                    )
                self.save(
                    os.path.join(cfg.model_dir, f"semgat_step_{total_steps}.pt"), total_steps
                )
                print(
                    f"[eval] step={total_steps} avg_return={eval_info['avg_return']:.2f} "
                    f"avg_length={eval_info['avg_length']:.1f} "
                    f"targets={eval_info['avg_targets_reached']:.2f}/{self.config.env.num_targets}"
                )

        # ── Final save ──
        final_path = os.path.join(cfg.model_dir, "semgat_final.pt")
        self.save(final_path, total_steps)
        if writer:
            writer.close()
        env.close()
        print(f"[SemGAT] training complete, saved {final_path}")

    def _bootstrap_value(self, global_obs: np.ndarray, base_env, active_agents) -> float:
        if not active_agents:
            return 0.0
        norm_global = self._normalize_global_obs(global_obs, update=False)
        global_t = torch.as_tensor(norm_global, dtype=torch.float32, device=self.device)

        # Build graph for current state
        edge_index, edge_type = self._build_graph(base_env)
        latent_z_batch = np.zeros((self.n_agents, self.latent_dim), dtype=np.float32)
        # Use zero latent for bootstrap (simplification)
        latent_t = torch.as_tensor(latent_z_batch, dtype=torch.float32, device=self.device)
        ei_t = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
        et_t = torch.as_tensor(edge_type, dtype=torch.long, device=self.device)

        with torch.no_grad():
            return float(self.critic(global_t, latent_t, ei_t, et_t).item())

    def _compute_returns_advantages(self, data: Dict[str, np.ndarray], next_value: float):
        cfg = self.config.train
        n = data["rewards"].shape[0]
        values = data["values"]
        masks = data["masks"]
        rewards = data["rewards"]
        dones = data["dones"]

        returns = np.zeros(n, dtype=np.float32)
        advantages = np.zeros((n, self.n_agents), dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            active = masks[t]
            active_count = max(float(active.sum()), 1.0)
            team_reward = float((rewards[t] * active).sum() / active_count)
            team_done = bool(dones[t].any())
            next_nonterminal = 0.0 if team_done else 1.0
            next_v = next_value if t == n - 1 else values[t + 1]
            delta = team_reward + cfg.gamma * next_v * next_nonterminal - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * gae
            returns[t] = gae + values[t]
            advantages[t] = gae * active

        return returns, advantages

    def _update(self, data: Dict[str, np.ndarray], next_value: float) -> Dict[str, float]:
        cfg = self.config.train
        returns, advantages = self._compute_returns_advantages(data, next_value)

        n_steps = data["rewards"].shape[0]
        obs_flat = data["obs"].reshape(-1, self.obs_dim)
        act_flat = data["actions"].reshape(-1, self.act_dim)
        latent_flat = data["latent_z"].reshape(-1, self.latent_dim)
        old_logp_flat = data["log_probs"].reshape(-1)
        adv_flat = advantages.reshape(-1)
        mask_flat = data["masks"].reshape(-1) > 0.5

        if mask_flat.sum() < 2:
            return {}

        # Filter active transitions
        obs_t = torch.as_tensor(obs_flat[mask_flat], dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act_flat[mask_flat], dtype=torch.float32, device=self.device)
        latent_t = torch.as_tensor(latent_flat[mask_flat], dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(old_logp_flat[mask_flat], dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv_flat[mask_flat], dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        global_t = torch.as_tensor(data["global_obs"], dtype=torch.float32, device=self.device)
        return_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_value_t = torch.as_tensor(data["values"], dtype=torch.float32, device=self.device)

        # Graph data
        edge_index_data = data["edge_index"]  # (n_steps, 2, max_edges)
        edge_type_data = data["edge_type"]  # (n_steps, max_edges)
        n_edges_data = data["n_edges"]  # (n_steps,)

        policy_losses = []
        value_losses = []
        entropies = []

        actor_indices = np.arange(obs_t.shape[0])
        critic_indices = np.arange(global_t.shape[0])

        for _ in range(cfg.n_epochs):
            # ── Actor update ──
            np.random.shuffle(actor_indices)
            for start in range(0, len(actor_indices), cfg.batch_size):
                idx = actor_indices[start : start + cfg.batch_size]
                # For actor update: construct self-loop-only graph for the mini-batch.
                # The batch contains agents from different timesteps, so cross-agent
                # edges are not meaningful. Self-loops preserve each node's own features.
                batch_n = idx.shape[0]
                ei_self = torch.stack([
                    torch.arange(batch_n, device=self.device),
                    torch.arange(batch_n, device=self.device),
                ])  # (2, batch_n)
                et_self = torch.zeros(batch_n, dtype=torch.long, device=self.device)

                new_logp, entropy = self.actor.evaluate_actions(
                    obs_t[idx], latent_t[idx], ei_self, et_self, act_t[idx]
                )
                ratio = (new_logp - old_logp_t[idx]).exp()
                unclipped = ratio * adv_t[idx]
                clipped = (
                    torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_t[idx]
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                loss = policy_loss - cfg.ent_coef * entropy.mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.actor_optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                entropies.append(float(entropy.mean().item()))

            # ── Critic update ──
            np.random.shuffle(critic_indices)
            for start in range(0, len(critic_indices), cfg.batch_size):
                idx = critic_indices[start : start + cfg.batch_size]

                # Flat-mode critic: no graph, use global obs directly
                # Zero latent to match batch size for critic forward
                latent_flat = torch.zeros(idx.shape[0], self.latent_dim, device=self.device)
                value = self.critic(global_t[idx], latent_flat, None, None)
                value_clipped = old_value_t[idx] + torch.clamp(
                    value - old_value_t[idx], -cfg.clip_range, cfg.clip_range
                )
                loss_unclipped = F.mse_loss(value, return_t[idx], reduction="none")
                loss_clipped = F.mse_loss(value_clipped, return_t[idx], reduction="none")
                value_loss = torch.max(loss_unclipped, loss_clipped).mean()

                self.critic_optimizer.zero_grad(set_to_none=True)
                (cfg.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_optimizer.step()

                value_losses.append(float(value_loss.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    # ── Evaluation ─────────────────────────────────────────────────────

    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        env = self._make_env()
        base_env = env.env
        returns = []
        lengths = []
        reached_list = []

        for ep in range(num_episodes):
            obs, info = env.reset(seed=self.config.train.seed + 10_000 + ep)
            done = False
            ep_return = 0.0
            ep_len = 0

            while not done:
                obs_batch = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
                for i, agent_id in enumerate(self.agent_ids):
                    if agent_id in env.agents:
                        obs_batch[i] = obs[agent_id]

                norm_obs = self._normalize_obs(obs_batch, update=False)
                latent_z = self._extract_latent(norm_obs)
                edge_index, edge_type = self._build_graph(base_env)

                obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device)
                latent_t = torch.as_tensor(latent_z, dtype=torch.float32, device=self.device)
                ei_t = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
                et_t = torch.as_tensor(edge_type, dtype=torch.long, device=self.device)

                with torch.no_grad():
                    actions_t, _, _ = self.actor.sample(obs_t, latent_t, ei_t, et_t, deterministic=True)
                action_arr = actions_t.cpu().numpy()
                actions = {
                    agent_id: action_arr[i] for i, agent_id in enumerate(self.agent_ids)
                }
                obs, rewards, terms, truncs, info = env.step(actions)
                ep_return += sum(rewards.values())
                ep_len += 1
                done = not env.agents

            returns.append(ep_return)
            lengths.append(ep_len)
            reached_list.append(int(np.sum(base_env._target_assigned)))

        env.close()
        return {
            "avg_return": float(np.mean(returns)) if returns else 0.0,
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
            "avg_targets_reached": float(np.mean(reached_list)) if reached_list else 0.0,
        }

    # ── Save / Load ────────────────────────────────────────────────────

    def save(self, path: str, step: int):
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "step": step,
            "config": {
                "env": asdict(self.config.env),
                "train": asdict(self.config.train),
                "semgat": asdict(self.config.semgat),
            },
            "dims": {
                "n_agents": self.n_agents,
                "obs_dim": self.obs_dim,
                "global_dim": self.global_dim,
                "act_dim": self.act_dim,
                "latent_dim": self.latent_dim,
            },
        }
        torch.save(payload, path)


# ── CLI ──────────────────────────────────────────────────────────────────


def build_config(args) -> FullConfig:
    cfg = FullConfig()
    cfg.env.num_drones = args.num_drones
    cfg.env.num_targets = args.num_targets
    cfg.env.num_obstacles = args.num_obstacles
    cfg.env.task_mode = args.task_mode
    cfg.env.aggregate_phy_steps = args.aggregate_phy_steps
    cfg.env.max_steps = args.max_episode_steps
    cfg.obs.max_num_drones = args.num_drones
    cfg.obs.max_num_targets = args.num_targets
    cfg.obs.max_num_obstacles = args.num_obstacles
    cfg.term.max_steps = args.max_episode_steps

    cfg.train.total_timesteps = args.total_steps
    cfg.train.seed = args.seed
    cfg.train.device = args.device
    cfg.train.learning_rate = args.lr
    cfg.train.n_steps = args.rollout_steps
    cfg.train.batch_size = args.batch_size
    cfg.train.n_epochs = args.epochs
    cfg.train.gamma = args.gamma
    cfg.train.gae_lambda = args.gae_lambda
    cfg.train.clip_range = args.clip_range
    cfg.train.ent_coef = args.ent_coef
    cfg.train.vf_coef = args.vf_coef
    cfg.train.max_grad_norm = args.max_grad_norm
    cfg.train.eval_freq = args.eval_freq
    cfg.train.n_eval_episodes = args.eval_episodes
    cfg.train.log_dir = args.log_dir
    cfg.train.model_dir = args.model_dir

    # SemGAT config
    cfg.semgat.use_vae_latent = not args.no_vae
    cfg.semgat.vae_checkpoint = args.vae_checkpoint
    cfg.vae.use_llm = args.use_llm
    cfg.vae.llm_model = args.llm_model
    cfg.vae.llm_api_key = args.llm_api_key
    cfg.semgat.gat_hidden_dim = args.gat_hidden
    cfg.semgat.gat_n_heads = args.gat_heads
    cfg.semgat.gat_n_layers = args.gat_layers
    cfg.semgat.use_semantic_reward = not args.no_semantic_reward
    cfg.semgat.rew_target_approach = args.rew_target
    cfg.semgat.rew_safe_distance = args.rew_safe

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Train SemGAT-MARL policy")

    # Environment
    parser.add_argument("--num_drones", type=int, default=4)
    parser.add_argument("--num_targets", type=int, default=4)
    parser.add_argument("--num_obstacles", type=int, default=10)
    parser.add_argument("--task_mode", choices=["reach", "delivery"], default="reach")
    parser.add_argument("--aggregate_phy_steps", type=int, default=20)
    parser.add_argument("--max_episode_steps", type=int, default=1000)

    # Training
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--rollout_steps", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.97)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # VAE
    parser.add_argument("--vae_checkpoint", type=str, default="models/semgat/vae_pretrained.pt")
    parser.add_argument("--no_vae", action="store_true", help="Run without VAE (ablation)")

    # LLM
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm_model", type=str, default="gpt-4o")
    parser.add_argument("--llm_api_key", type=str, default="")

    # GAT architecture
    parser.add_argument("--gat_hidden", type=int, default=64)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--gat_layers", type=int, default=2)

    # Semantic reward
    parser.add_argument("--no_semantic_reward", action="store_true")
    parser.add_argument("--rew_target", type=float, default=2.0)
    parser.add_argument("--rew_safe", type=float, default=0.5)

    # Output
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_dir", type=str, default="logs/semgat")
    parser.add_argument("--model_dir", type=str, default="models/semgat")
    parser.add_argument("--save_config", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    os.makedirs(cfg.train.model_dir, exist_ok=True)

    if args.save_config:
        config_path = os.path.join(cfg.train.log_dir, "semgat_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {"env": asdict(cfg.env), "train": asdict(cfg.train), "semgat": asdict(cfg.semgat)},
                f,
                indent=2,
            )

    trainer = SemGATMAPPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
