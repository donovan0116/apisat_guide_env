"""
MAPPO training entry point for the quadrotor delivery environment.

This is a self-contained CTDE implementation:
    - decentralized shared actor: local agent observation -> continuous action
    - centralized critic: global environment state -> scalar team value

Example:
    python3 scripts/mappo_train.py --num_drones 4 --num_targets 4 --total_steps 200000
    python3 scripts/mappo_train.py --task_mode delivery --num_obstacles 20 --device cpu
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs import ParallelQuadrotorDelivery
from utils.config import FullConfig

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
TANH_EPS = 1e-6


class RunningMeanStd:
    """Running mean and standard deviation for online normalization."""

    def __init__(self, shape: tuple, eps: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = eps

    def update(self, x: np.ndarray):
        """Update running statistics with a batch of samples."""
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


def mlp(
    in_dim: int,
    out_dim: int,
    hidden: int,
    n_layers: int,
    activation: str = "relu",
    orthogonal_init: bool = True,
) -> nn.Sequential:
    layers = []
    last_dim = in_dim
    act_cls = nn.ReLU if activation == "relu" else nn.Tanh
    for _ in range(n_layers):
        layers.append(nn.Linear(last_dim, hidden))
        if orthogonal_init:
            nn.init.orthogonal_(layers[-1].weight, gain=np.sqrt(2))
            nn.init.constant_(layers[-1].bias, 0.0)
        layers.append(act_cls())
        last_dim = hidden
    layers.append(nn.Linear(last_dim, out_dim))
    if orthogonal_init:
        # Last layer: small weight for value, small/zero for policy mean
        nn.init.orthogonal_(layers[-1].weight, gain=0.01)
        nn.init.constant_(layers[-1].bias, 0.0)
    return nn.Sequential(*layers)


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + TANH_EPS, 1.0 - TANH_EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class SquashedGaussianActor(nn.Module):
    """Parameter-shared actor for all drones."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, n_layers: int):
        super().__init__()
        self.net = mlp(
            obs_dim, hidden, hidden, n_layers, activation="relu", orthogonal_init=True
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0.0)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.constant_(self.log_std.bias, -0.5)  # std ~0.6 at init for exploration

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(dist, pre_tanh, action)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        clipped_actions = torch.clamp(actions, -1.0 + TANH_EPS, 1.0 - TANH_EPS)
        pre_tanh = atanh(clipped_actions)
        log_prob = self._log_prob_from_pre_tanh(dist, pre_tanh, clipped_actions)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    @staticmethod
    def _log_prob_from_pre_tanh(
        dist: Normal, pre_tanh: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        correction = torch.log(1.0 - action.pow(2) + TANH_EPS)
        return (dist.log_prob(pre_tanh) - correction).sum(dim=-1)


class CentralCritic(nn.Module):
    """Centralized value function that consumes global observations."""

    def __init__(self, global_dim: int, hidden: int, n_layers: int):
        super().__init__()
        self.net = mlp(
            global_dim, 1, hidden, n_layers, activation="relu", orthogonal_init=True
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)


class RolloutBuffer:
    def __init__(
        self, n_steps: int, n_agents: int, obs_dim: int, global_dim: int, act_dim: int
    ):
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.global_dim = global_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        self.ptr = 0
        self.obs = np.zeros(
            (self.n_steps, self.n_agents, self.obs_dim), dtype=np.float32
        )
        self.global_obs = np.zeros((self.n_steps, self.global_dim), dtype=np.float32)
        self.actions = np.zeros(
            (self.n_steps, self.n_agents, self.act_dim), dtype=np.float32
        )
        self.log_probs = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)
        self.values = np.zeros(self.n_steps, dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_agents), dtype=np.bool_)
        self.masks = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)

    def add(self, obs, global_obs, actions, log_probs, rewards, value, dones, masks):
        if self.ptr >= self.n_steps:
            raise RuntimeError("RolloutBuffer overflow")
        self.obs[self.ptr] = obs
        self.global_obs[self.ptr] = global_obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = value
        self.dones[self.ptr] = dones
        self.masks[self.ptr] = masks
        self.ptr += 1

    def data(self):
        return {
            "obs": self.obs[: self.ptr],
            "global_obs": self.global_obs[: self.ptr],
            "actions": self.actions[: self.ptr],
            "log_probs": self.log_probs[: self.ptr],
            "rewards": self.rewards[: self.ptr],
            "values": self.values[: self.ptr],
            "dones": self.dones[: self.ptr],
            "masks": self.masks[: self.ptr],
        }


class MAPPOTrainer:
    def __init__(self, config: FullConfig):
        self.config = config
        self.device = torch.device(config.train.device)
        torch.manual_seed(config.train.seed)
        np.random.seed(config.train.seed)

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
        self.actor = SquashedGaussianActor(
            self.obs_dim,
            self.act_dim,
            train_cfg.policy_hidden_size,
            train_cfg.policy_n_layers,
        ).to(self.device)
        self.critic = CentralCritic(
            self.global_dim,
            train_cfg.value_hidden_size,
            train_cfg.value_n_layers,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=train_cfg.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=train_cfg.learning_rate
        )
        self.buffer = RolloutBuffer(
            train_cfg.n_steps,
            self.n_agents,
            self.obs_dim,
            self.global_dim,
            self.act_dim,
        )

        # Online normalization for stability
        self.obs_rms = RunningMeanStd((self.obs_dim,))
        self.global_obs_rms = RunningMeanStd((self.global_dim,))

    def _normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """Normalize local observations using running statistics."""
        flat = obs.reshape(-1, self.obs_dim)
        if update:
            self.obs_rms.update(flat)
        norm = self.obs_rms.normalize(flat)
        return norm.reshape(obs.shape)

    def _normalize_global_obs(
        self, global_obs: np.ndarray, update: bool = True
    ) -> np.ndarray:
        """Normalize global observations using running statistics."""
        if global_obs.ndim == 1:
            g = global_obs[None, :]
        else:
            g = global_obs
        if update:
            self.global_obs_rms.update(g)
        return self.global_obs_rms.normalize(g).reshape(global_obs.shape)

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

    def train(self):
        cfg = self.config.train
        env = self._make_env()
        os.makedirs(cfg.model_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        writer = SummaryWriter(cfg.log_dir) if SummaryWriter else None
        if writer is None:
            print(
                "[MAPPO] TensorBoard unavailable. Install tensorboard to enable logging."
            )

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
            f"[MAPPO] training {cfg.total_timesteps} env steps | "
            f"{self.n_agents} drones | obs={self.obs_dim} global={self.global_dim} act={self.act_dim}"
        )

        while total_steps < cfg.total_timesteps:
            self.buffer.reset()

            for _ in range(cfg.n_steps):
                active_agents = set(env.agents)
                obs_batch = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
                actions_batch = np.zeros(
                    (self.n_agents, self.act_dim), dtype=np.float32
                )
                log_probs = np.zeros(self.n_agents, dtype=np.float32)
                masks = np.zeros(self.n_agents, dtype=np.float32)

                for i, agent_id in enumerate(self.agent_ids):
                    if agent_id not in active_agents:
                        continue
                    obs_batch[i] = obs_dict[agent_id]
                    masks[i] = 1.0

                # Normalize inputs using running statistics
                norm_obs_batch = self._normalize_obs(obs_batch)
                norm_global_obs = self._normalize_global_obs(global_obs)

                obs_tensor = torch.as_tensor(
                    norm_obs_batch, dtype=torch.float32, device=self.device
                )
                global_tensor = torch.as_tensor(
                    norm_global_obs, dtype=torch.float32, device=self.device
                )

                with torch.no_grad():
                    action_tensor, log_prob_tensor, _ = self.actor.sample(obs_tensor)
                    value = self.critic(global_tensor.unsqueeze(0)).item()

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

                self.buffer.add(
                    norm_obs_batch,
                    norm_global_obs,
                    actions_batch,
                    log_probs,
                    reward_batch,
                    value,
                    done_batch,
                    masks,
                )

                total_steps += 1
                episode_len += 1
                episode_reward += float((reward_batch * masks).sum())

                obs_dict = next_obs
                global_obs = next_info["global_obs"].astype(np.float32)

                if not env.agents:
                    reward_window.append(episode_reward)
                    episode_idx += 1
                    if episode_idx % cfg.n_eval_episodes == 0:
                        avg = np.mean(reward_window) if reward_window else 0.0
                        fps = int(total_steps / max(time.time() - start_time, 1e-6))
                        if writer:
                            writer.add_scalar(
                                "rollout/episode_return", episode_reward, total_steps
                            )
                            writer.add_scalar(
                                "rollout/episode_return_mean_100", avg, total_steps
                            )
                            writer.add_scalar(
                                "rollout/episode_length", episode_len, total_steps
                            )
                            writer.add_scalar("time/fps", fps, total_steps)
                        print(
                            f"step={total_steps:>8d} update={updates:>5d} "
                            f"episode={episode_idx:>5d} avg_return={avg:>9.2f} "
                            f"last_len={episode_len:>4d} fps={fps:>4d}"
                        )
                    obs_dict, info = env.reset()
                    global_obs = info["global_obs"].astype(np.float32)
                    episode_reward = 0.0
                    episode_len = 0

                if total_steps >= cfg.total_timesteps:
                    break

            if self.buffer.ptr > 1:
                next_value = self._bootstrap_value(global_obs, env.agents)
                update_info = self._update(self.buffer.data(), next_value)
                updates += 1

                # Linear learning-rate annealing
                progress = min(1.0, total_steps / cfg.total_timesteps)
                lr = cfg.learning_rate * (1.0 - progress)
                for pg in self.actor_optimizer.param_groups:
                    pg["lr"] = lr
                for pg in self.critic_optimizer.param_groups:
                    pg["lr"] = lr

                if writer and update_info:
                    writer.add_scalar(
                        "train/actor_loss", update_info["policy_loss"], total_steps
                    )
                    writer.add_scalar(
                        "train/critic_loss", update_info["value_loss"], total_steps
                    )
                    writer.add_scalar(
                        "train/entropy", update_info["entropy"], total_steps
                    )
                    writer.add_scalar("train/learning_rate", lr, total_steps)
            else:
                update_info = {}

            if updates > 0 and (total_steps % cfg.eval_freq < cfg.n_steps):
                eval_info = self.evaluate(num_episodes=cfg.n_eval_episodes)
                if writer:
                    writer.add_scalar(
                        "eval/avg_return", eval_info["avg_return"], total_steps
                    )
                    writer.add_scalar(
                        "eval/avg_length", eval_info["avg_length"], total_steps
                    )
                    writer.add_scalar(
                        "eval/avg_targets_reached",
                        eval_info["avg_targets_reached"],
                        total_steps,
                    )
                self.save(
                    os.path.join(cfg.model_dir, f"mappo_step_{total_steps}.pt"),
                    total_steps,
                )
                print(
                    f"[eval] step={total_steps} avg_return={eval_info['avg_return']:.2f} "
                    f"avg_length={eval_info['avg_length']:.1f} "
                    f"targets={eval_info['avg_targets_reached']:.2f}/{self.config.env.num_targets} "
                    f"policy_loss={update_info.get('policy_loss', 0.0):.4f} "
                    f"value_loss={update_info.get('value_loss', 0.0):.4f}"
                )

        final_path = os.path.join(cfg.model_dir, "mappo_final.pt")
        self.save(final_path, total_steps)
        if writer:
            writer.close()
        env.close()
        print(f"[MAPPO] complete, saved {final_path}")

    def _bootstrap_value(self, global_obs: np.ndarray, active_agents) -> float:
        if not active_agents:
            return 0.0
        norm_global = self._normalize_global_obs(global_obs, update=False)
        global_tensor = torch.as_tensor(
            norm_global, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            return float(self.critic(global_tensor.unsqueeze(0)).item())

    def _compute_returns_advantages(
        self, data: Dict[str, np.ndarray], next_value: float
    ):
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

    def _update(
        self, data: Dict[str, np.ndarray], next_value: float
    ) -> Dict[str, float]:
        cfg = self.config.train
        returns, advantages = self._compute_returns_advantages(data, next_value)

        obs_flat = data["obs"].reshape(-1, self.obs_dim)
        act_flat = data["actions"].reshape(-1, self.act_dim)
        old_logp_flat = data["log_probs"].reshape(-1)
        adv_flat = advantages.reshape(-1)
        mask_flat = data["masks"].reshape(-1) > 0.5

        if mask_flat.sum() < 2:
            return {}

        obs_t = torch.as_tensor(
            obs_flat[mask_flat], dtype=torch.float32, device=self.device
        )
        act_t = torch.as_tensor(
            act_flat[mask_flat], dtype=torch.float32, device=self.device
        )
        old_logp_t = torch.as_tensor(
            old_logp_flat[mask_flat], dtype=torch.float32, device=self.device
        )
        adv_t = torch.as_tensor(
            adv_flat[mask_flat], dtype=torch.float32, device=self.device
        )
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        global_t = torch.as_tensor(
            data["global_obs"], dtype=torch.float32, device=self.device
        )
        return_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_value_t = torch.as_tensor(
            data["values"], dtype=torch.float32, device=self.device
        )

        policy_losses = []
        value_losses = []
        entropies = []

        actor_indices = np.arange(obs_t.shape[0])
        critic_indices = np.arange(global_t.shape[0])

        for _ in range(cfg.n_epochs):
            np.random.shuffle(actor_indices)
            for start in range(0, len(actor_indices), cfg.batch_size):
                idx = actor_indices[start : start + cfg.batch_size]
                new_logp, entropy = self.actor.evaluate_actions(obs_t[idx], act_t[idx])
                ratio = (new_logp - old_logp_t[idx]).exp()
                unclipped = ratio * adv_t[idx]
                clipped = (
                    torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                    * adv_t[idx]
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                loss = policy_loss - cfg.ent_coef * entropy.mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.actor_optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                entropies.append(float(entropy.mean().item()))

            np.random.shuffle(critic_indices)
            for start in range(0, len(critic_indices), cfg.batch_size):
                idx = critic_indices[start : start + cfg.batch_size]
                value = self.critic(global_t[idx])
                value_clipped = old_value_t[idx] + torch.clamp(
                    value - old_value_t[idx], -cfg.clip_range, cfg.clip_range
                )
                loss_unclipped = F.mse_loss(value, return_t[idx], reduction="none")
                loss_clipped = F.mse_loss(
                    value_clipped, return_t[idx], reduction="none"
                )
                value_loss = 0.5 * torch.max(loss_unclipped, loss_clipped).mean()

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

    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        env = self._make_env()
        returns = []
        lengths = []
        reached = []

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

                obs_t = torch.as_tensor(
                    self._normalize_obs(obs_batch, update=False),
                    dtype=torch.float32,
                    device=self.device,
                )
                with torch.no_grad():
                    actions_t, _, _ = self.actor.sample(obs_t, deterministic=True)
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
            reached.append(int(np.sum(env.env._target_assigned)))

        env.close()
        return {
            "avg_return": float(np.mean(returns)) if returns else 0.0,
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
            "avg_targets_reached": float(np.mean(reached)) if reached else 0.0,
        }

    def save(self, path: str, step: int):
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "step": step,
            "config": {
                "env": asdict(self.config.env),
                "train": asdict(self.config.train),
            },
            "dims": {
                "n_agents": self.n_agents,
                "obs_dim": self.obs_dim,
                "global_dim": self.global_dim,
                "act_dim": self.act_dim,
            },
        }
        torch.save(payload, path)


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
    cfg.train.policy_hidden_size = args.hidden_size
    cfg.train.value_hidden_size = args.hidden_size
    cfg.train.policy_n_layers = args.layers
    cfg.train.value_n_layers = args.layers
    cfg.train.eval_freq = args.eval_freq
    cfg.train.n_eval_episodes = args.eval_episodes
    cfg.train.log_dir = args.log_dir
    cfg.train.model_dir = args.model_dir
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MAPPO baseline.")
    parser.add_argument("--num_drones", type=int, default=4)
    parser.add_argument("--num_targets", type=int, default=4)
    parser.add_argument("--num_obstacles", type=int, default=10)
    parser.add_argument("--task_mode", choices=["reach", "delivery"], default="reach")
    parser.add_argument("--aggregate_phy_steps", type=int, default=1)
    parser.add_argument("--max_episode_steps", type=int, default=2000)

    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--rollout_steps", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)

    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_dir", type=str, default="logs/mappo")
    parser.add_argument("--model_dir", type=str, default="models/mappo")
    parser.add_argument("--save_config", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    os.makedirs(cfg.train.model_dir, exist_ok=True)

    if args.save_config:
        config_path = os.path.join(cfg.train.log_dir, "mappo_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"env": asdict(cfg.env), "train": asdict(cfg.train)}, f, indent=2)

    trainer = MAPPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
