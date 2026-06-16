"""
CTDE Trainer: Centralized Training with Decentralized Execution (MAPPO-style).

Architecture:
    - Actor Network:  local_obs -> action (per drone, parameter-shared)
    - Critic Network: global_obs -> value (centralized, sees full state)

Training:
    - Collect rollouts from all agents using the shared actor
    - Update actor with PPO clipped objective (per-agent advantages)
    - Update critic with TD/GAM returns from centralized viewpoint

Usage:
    python scripts/ctde_trainer.py --num_drones 4 --num_targets 4
    python scripts/ctde_trainer.py --num_drones 6 --task_mode delivery
"""

import argparse
import os
import sys
from collections import deque
from typing import List, Dict, Tuple

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


# ============================================================
#  Neural Networks
# ============================================================
class Actor(nn.Module):
    """Shared actor: local_obs -> action mean + log_std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(obs)
        mean = self.mean_head(x)
        std = torch.exp(self.log_std.clamp(-10, 2))
        return mean, std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Centralized critic: global_obs -> scalar value."""

    def __init__(self, obs_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ============================================================
#  Rollout Buffer
# ============================================================
class RolloutBuffer:
    def __init__(self, n_agents: int, obs_dim: int, global_dim: int, act_dim: int,
                 capacity: int):
        self.n_agents = n_agents
        self.capacity = capacity
        self.ptr = 0

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.global_obs = np.zeros((capacity, global_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.log_probs = np.zeros((capacity, n_agents), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity, n_agents), dtype=np.bool_)
        self.masks = np.zeros((capacity, n_agents), dtype=np.float32)  # 1=active

    def add(self, obs, global_obs, actions, rewards, log_probs, value, dones, masks):
        if self.ptr >= self.capacity:
            return
        self.obs[self.ptr] = obs
        self.global_obs[self.ptr] = global_obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = value
        self.dones[self.ptr] = dones
        self.masks[self.ptr] = masks
        self.ptr += 1

    def clear(self):
        self.ptr = 0

    def get_data(self):
        return {k: v[:self.ptr] for k, v in self.__dict__.items()
                if isinstance(v, np.ndarray)}


# ============================================================
#  Trainer
# ============================================================
class CTDETrainer:
    def __init__(self, config: FullConfig):
        self.config = config
        self.device = torch.device(config.train.device)

        # Build env to get dimensions
        temp_env = ParallelQuadrotorDelivery(
            num_drones=config.env.num_drones,
            num_targets=config.env.num_targets,
            num_obstacles=config.env.num_obstacles,
            task_mode=config.env.task_mode,
            bounds=config.env.bounds_array,
            quad_params=config.quad,
            obs_config=config.obs,
            act_config=config.act,
            reward_config=config.reward,
            term_config=config.term,
        )
        agent_id = temp_env.possible_agents[0]
        self.obs_dim = temp_env.observation_space(agent_id).shape[0]
        self.act_dim = temp_env.action_space(agent_id).shape[0]
        self.n_agents = config.env.num_drones
        obs_dict, info = temp_env.reset()
        self.global_dim = info["global_obs"].shape[0]
        temp_env.close()

        self.actor = Actor(self.obs_dim, self.act_dim,
                           config.train.policy_hidden_size,
                           config.train.policy_n_layers).to(self.device)
        self.critic = Critic(self.global_dim,
                             config.train.value_hidden_size,
                             config.train.value_n_layers).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(),
                                            lr=config.train.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=config.train.learning_rate)

        self.buffer = RolloutBuffer(
            self.n_agents, self.obs_dim, self.global_dim, self.act_dim,
            config.train.n_steps,
        )

    def train(self):
        config = self.config
        os.makedirs(config.train.log_dir, exist_ok=True)
        os.makedirs(config.train.model_dir, exist_ok=True)
        writer = SummaryWriter(config.train.log_dir) if SummaryWriter else None
        if writer is None:
            print("[CTDE] TensorBoard unavailable. Install tensorboard to enable logging.")
        env = ParallelQuadrotorDelivery(
            num_drones=config.env.num_drones,
            num_targets=config.env.num_targets,
            num_obstacles=config.env.num_obstacles,
            task_mode=config.env.task_mode,
            bounds=config.env.bounds_array,
            quad_params=config.quad,
            obs_config=config.obs,
            act_config=config.act,
            reward_config=config.reward,
            term_config=config.term,
        )

        total_steps = 0
        episode = 0
        episode_rewards = deque(maxlen=100)
        current_episode_reward = 0.0
        current_episode_len = 0

        obs_dict, info_dict = env.reset()
        global_obs = info_dict["global_obs"]

        while total_steps < config.train.total_timesteps:
            # -- Collect rollout --
            self.buffer.clear()
            for _ in range(config.train.n_steps):
                # Gather batch
                batch_obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
                batch_acts = np.zeros((self.n_agents, self.act_dim), dtype=np.float32)
                batch_logps = np.zeros(self.n_agents, dtype=np.float32)
                masks = np.ones(self.n_agents, dtype=np.float32)

                for j, agent_id in enumerate(env.possible_agents):
                    if agent_id not in env.agents:
                        masks[j] = 0.0
                        continue
                    obs_t = torch.as_tensor(obs_dict[agent_id], device=self.device).float()
                    with torch.no_grad():
                        act_t, logp_t, _ = self.actor.get_action(obs_t)
                    batch_obs[j] = obs_dict[agent_id]
                    batch_acts[j] = act_t.cpu().numpy()
                    batch_logps[j] = logp_t.item()

                # Critic value
                global_t = torch.as_tensor(global_obs, device=self.device).float()
                with torch.no_grad():
                    value = self.critic(global_t).item()

                # Env step
                actions = {env.possible_agents[j]: batch_acts[j] for j in range(self.n_agents)}
                next_obs, rewards_dict, terms, truncs, next_info = env.step(actions)

                rewards = np.array([
                    rewards_dict.get(env.possible_agents[j], 0.0)
                    for j in range(self.n_agents)
                ], dtype=np.float32)
                dones = np.array([
                    terms.get(env.possible_agents[j], False) or
                    truncs.get(env.possible_agents[j], False)
                    for j in range(self.n_agents)
                ], dtype=np.bool_)

                self.buffer.add(batch_obs, global_obs, batch_acts,
                                rewards, batch_logps, value, dones, masks)

                obs_dict = next_obs
                global_obs = next_info.get("global_obs", global_obs)
                total_steps += 1
                current_episode_reward += float((rewards * masks).sum())
                current_episode_len += 1

                if not env.agents:
                    obs_dict, info_dict = env.reset()
                    global_obs = info_dict["global_obs"]
                    episode += 1
                    episode_rewards.append(current_episode_reward)
                    if writer:
                        writer.add_scalar("rollout/episode_reward", current_episode_reward, total_steps)
                        writer.add_scalar("rollout/episode_reward_mean_100", np.mean(episode_rewards), total_steps)
                        writer.add_scalar("rollout/episode_length", current_episode_len, total_steps)
                    current_episode_reward = 0.0
                    current_episode_len = 0
                    break

            # -- PPO Update --
            if self.buffer.ptr < 2:
                continue

            data = self.buffer.get_data()
            update_info = self._update(data)
            if writer and update_info:
                writer.add_scalar("train/actor_loss", update_info["actor_loss"], total_steps)
                writer.add_scalar("train/critic_loss", update_info["critic_loss"], total_steps)
                writer.add_scalar("train/entropy", update_info["entropy"], total_steps)

            # Logging
            if total_steps % 5000 == 0 and episode_rewards:
                avg_reward = np.mean(episode_rewards)
                print(f"Step {total_steps:>8d} | Ep {episode:>4d} | "
                      f"Avg Reward: {avg_reward:>8.1f}")

            # Checkpoint
            if total_steps % 20000 == 0 or total_steps >= config.train.total_timesteps:
                ckpt_dir = config.train.model_dir
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "step": total_steps,
                }, os.path.join(ckpt_dir, f"ctde_step_{total_steps}.pt"))

        if writer:
            writer.close()
        env.close()
        print(f"[CTDE] Training complete. Final avg reward: "
              f"{np.mean(episode_rewards):.1f}" if episode_rewards else "")

    def _update(self, data: dict):
        config = self.config.train
        n = data["obs"].shape[0]  # timesteps collected

        # Compute advantages using GAE
        obs_np = data["global_obs"]
        rewards_np = data["rewards"]
        dones_np = data["dones"]
        masks_np = data["masks"]
        values_np = data["values"]

        # Next value for bootstrapping (last global obs)
        global_last = torch.as_tensor(obs_np[-1:], device=self.device).float()
        with torch.no_grad():
            next_value = self.critic(global_last).item()

        returns = np.zeros(n)
        advantages = np.zeros((n, self.n_agents))
        gae = np.zeros(self.n_agents)

        for t in reversed(range(n)):
            mask_t = masks_np[t]
            team_reward = (rewards_np[t] * mask_t).sum() / (mask_t.sum() + 1e-8)
            team_done = dones_np[t].any()
            next_val = next_value if t == n - 1 else values_np[t + 1]

            delta = team_reward + config.gamma * next_val * (1 - team_done) - values_np[t]
            gae = delta + config.gamma * config.gae_lambda * (1 - team_done) * gae
            advantages[t] = gae * mask_t  # broadcast per-agent
            returns[t] = advantages[t].mean() + values_np[t]

            next_value = values_np[t]

        # Flatten for actor (per-agent transitions)
        obs_flat = data["obs"].reshape(-1, self.obs_dim)
        acts_flat = data["actions"].reshape(-1, self.act_dim)
        logps_flat = data["log_probs"].reshape(-1)
        advs_flat = advantages.reshape(-1)
        masks_flat = masks_np.reshape(-1)

        # Only update actor on active transitions
        active = masks_flat > 0.5
        n_active = active.sum()
        if n_active < 2:
            return {}

        obs_t = torch.as_tensor(obs_flat[active], device=self.device).float()
        acts_t = torch.as_tensor(acts_flat[active], device=self.device).float()
        logps_t = torch.as_tensor(logps_flat[active], device=self.device).float()
        advs_t = torch.as_tensor(advs_flat[active], device=self.device).float()

        # Normalize advantages (per-agent)
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        # --- Actor update ---
        m = obs_t.shape[0]
        indices = np.arange(m)
        actor_losses = []
        critic_losses = []
        entropies = []

        for _ in range(config.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, m, config.batch_size):
                end = min(start + config.batch_size, m)
                idx = indices[start:end]

                new_logp, entropy = self.actor.evaluate(obs_t[idx], acts_t[idx])
                ratio = torch.exp(new_logp - logps_t[idx])
                surr1 = ratio * advs_t[idx]
                surr2 = torch.clamp(ratio, 1 - config.clip_range,
                                    1 + config.clip_range) * advs_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                actor_loss = policy_loss - config.ent_coef * entropy.mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), config.max_grad_norm)
                self.actor_optim.step()
                actor_losses.append(float(policy_loss.item()))
                entropies.append(float(entropy.mean().item()))

        # --- Critic update (per-timestep, full global state) ---
        global_t = torch.as_tensor(data["global_obs"], device=self.device).float()
        rets_t = torch.as_tensor(returns, device=self.device).float()

        n_t = global_t.shape[0]
        t_indices = np.arange(n_t)

        for _ in range(config.n_epochs):
            np.random.shuffle(t_indices)
            for start in range(0, n_t, config.batch_size):
                end = min(start + config.batch_size, n_t)
                idx = t_indices[start:end]

                values = self.critic(global_t[idx])
                critic_loss = F.mse_loss(values, rets_t[idx])

                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), config.max_grad_norm)
                self.critic_optim.step()
                critic_losses.append(float(critic_loss.item()))

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CTDE MAPPO training")
    parser.add_argument("--num_drones", type=int, default=4)
    parser.add_argument("--num_targets", type=int, default=4)
    parser.add_argument("--num_obstacles", type=int, default=10)
    parser.add_argument("--task_mode", type=str, default="reach",
                        choices=["reach", "delivery"])
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_dir", type=str, default="./logs/ctde")
    parser.add_argument("--model_dir", type=str, default="./models")
    args = parser.parse_args()

    config = FullConfig()
    config.env.num_drones = args.num_drones
    config.env.num_targets = args.num_targets
    config.env.num_obstacles = args.num_obstacles
    config.env.task_mode = args.task_mode
    config.train.total_timesteps = args.total_steps
    config.train.seed = args.seed
    config.train.device = args.device
    config.train.n_steps = 2048
    config.train.log_dir = args.log_dir
    config.train.model_dir = args.model_dir

    trainer = CTDETrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
