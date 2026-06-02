"""
Training script for multi-UAV delivery task assignment.

Supports:
    ppo  - Centralized PPO: single policy controls all drones (flattened obs)
    ippo - Independent PPO: parameter sharing, each drone runs its own policy

Usage:
    python scripts/train.py --algo ppo --num_drones 4 --num_targets 4
    python scripts/train.py --algo ippo --num_drones 6 --task_mode delivery
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.dynamics import QuadrotorParams
from core.state import ObsConfig
from core.action import ActionConfig
from core.reward import RewardConfig
from core.termination import TerminationConfig
from utils.config import FullConfig
from envs import QuadrotorDeliveryEnv, ParallelQuadrotorDelivery, make_sb3_env


# ============================================================
#  Centralized PPO
# ============================================================
def train_centralized_ppo(config: FullConfig):
    """Train a single policy that controls all drones simultaneously."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError:
        print("Stable-Baselines3 required. Run: pip install stable-baselines3")
        return

    def _make():
        env = QuadrotorDeliveryEnv(
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
            aggregate_phy_steps=config.env.aggregate_phy_steps,
            seed=config.train.seed,
        )
        return make_sb3_env(env)

    n_envs = config.train.n_envs
    env = DummyVecEnv([_make for _ in range(n_envs)])
    # Note: VecNormalize removed due to shape compatibility;
    # normalize observations via ObsConfig.normalize=True instead.

    os.makedirs(config.train.log_dir, exist_ok=True)
    os.makedirs(config.train.model_dir, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.train.learning_rate,
        n_steps=config.train.n_steps,
        batch_size=config.train.batch_size,
        n_epochs=config.train.n_epochs,
        gamma=config.train.gamma,
        gae_lambda=config.train.gae_lambda,
        clip_range=config.train.clip_range,
        ent_coef=config.train.ent_coef,
        vf_coef=config.train.vf_coef,
        max_grad_norm=config.train.max_grad_norm,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[config.train.policy_hidden_size] * config.train.policy_n_layers,
                vf=[config.train.value_hidden_size] * config.train.value_n_layers,
            )
        ),
        verbose=1,
        tensorboard_log=config.train.log_dir,
        seed=config.train.seed,
        device=config.train.device,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=config.train.eval_freq // n_envs,
        save_path=config.train.model_dir,
        name_prefix="ppo_centralized",
    )

    print(f"[Centralized PPO] Training {config.train.total_timesteps} steps "
          f"({n_envs} envs, {config.env.num_drones} drones, "
          f"{config.env.num_targets} targets)")
    model.learn(
        total_timesteps=config.train.total_timesteps,
        callback=checkpoint_cb,
        tb_log_name="quadrotor_delivery",
        progress_bar=True,
    )

    model.save(os.path.join(config.train.model_dir, "ppo_centralized_final"))
    env.close()
    print("[Centralized PPO] Training complete.")


# ============================================================
#  Independent PPO  (parameter sharing via same Policy)
# ============================================================
def train_independent_ppo(config: FullConfig):
    """
    Train independent policies with parameter sharing.

    Uses the PettingZoo wrapper. A single SB3 PPO model is cloned per agent;
    all share weights. Training loop collects rollouts from all agents,
    concatenates them, and updates the shared policy.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import get_schedule_fn
        from stable_baselines3.common.buffers import RolloutBuffer
        from stable_baselines3.common.callbacks import BaseCallback
        import torch
        import torch.nn as nn
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError:
        print("Stable-Baselines3 + PyTorch required. Run: pip install stable-baselines3 torch")
        return

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
        aggregate_phy_steps=config.env.aggregate_phy_steps,
        seed=config.train.seed,
    )

    agent_ids = env.possible_agents
    obs_space = env.observation_space(agent_ids[0])
    act_space = env.action_space(agent_ids[0])

    os.makedirs(config.train.model_dir, exist_ok=True)

    # --- Manual parameter-sharing training loop ---
    print(f"[Independent PPO] Training {config.train.total_timesteps} steps "
          f"({config.env.num_drones} drones, {config.env.num_targets} targets)")

    device = torch.device(config.train.device)

    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.ppo import PPO

    # Create a single PPO model (shared parameters for all agents)
    class SharedPolicyEnv(gym.Env):
        """Thin wrapper so SB3 can construct the policy."""

        def __init__(self):
            self.observation_space = obs_space
            self.action_space = act_space

    dummy_env = SharedPolicyEnv()

    model = PPO(
        "MlpPolicy",
        dummy_env,
        learning_rate=config.train.learning_rate,
        n_steps=config.train.n_steps,
        batch_size=config.train.batch_size,
        n_epochs=config.train.n_epochs,
        gamma=config.train.gamma,
        gae_lambda=config.train.gae_lambda,
        clip_range=config.train.clip_range,
        ent_coef=config.train.ent_coef,
        vf_coef=config.train.vf_coef,
        max_grad_norm=config.train.max_grad_norm,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[config.train.policy_hidden_size] * config.train.policy_n_layers,
                vf=[config.train.value_hidden_size] * config.train.value_n_layers,
            )
        ),
        verbose=1,
        seed=config.train.seed,
        device=config.train.device,
    )

    n_steps = config.train.n_steps
    total_steps = 0
    episode = 0
    best_reward = -np.inf

    obs_dict, info_dict = env.reset()

    # Training loop
    while total_steps < config.train.total_timesteps:
        # Collect rollout from all agents
        obs_list = [[] for _ in agent_ids]
        act_list = [[] for _ in agent_ids]
        rew_list = [[] for _ in agent_ids]
        done_list = [[] for _ in agent_ids]
        val_list = [[] for _ in agent_ids]
        logp_list = [[] for _ in agent_ids]

        for step in range(n_steps):
            actions = {}
            for agent_id in agent_ids:
                i = int(agent_id.split("_")[1])
                obs_t = torch.as_tensor(obs_dict[agent_id], device=device).float().unsqueeze(0)
                with torch.no_grad():
                    dist = model.policy.get_distribution(obs_t)
                    act_t = dist.sample()
                    logp_t = dist.log_prob(act_t)
                    val_t = model.policy.predict_values(obs_t)
                actions[agent_id] = act_t.cpu().numpy().flatten()
                obs_list[i].append(obs_dict[agent_id])
                act_list[i].append(actions[agent_id])
                val_list[i].append(val_t.item())
                logp_list[i].append(logp_t.item())

            # Step environment
            next_obs, rewards, terms, truncs, infos = env.step(actions)

            for agent_id in agent_ids:
                i = int(agent_id.split("_")[1])
                rew_list[i].append(rewards.get(agent_id, 0.0))
                done_list[i].append(terms.get(agent_id, False) or truncs.get(agent_id, False))

            obs_dict = next_obs
            total_steps += 1

            # Reset if all done
            if not env.agents:
                obs_dict, info_dict = env.reset()
                episode += 1
                total_rew = sum(rew_list[j][-1] if rew_list[j] else 0 for j in range(len(agent_ids)))
                if total_steps % 5000 == 0:
                    print(f"  Step {total_steps} | Episode {episode} | "
                          f"Reward {total_rew:.1f}")

        # --- PPO update (shared policy) ---
        # Stack all agents' data
        all_obs = []
        all_acts = []
        all_rews = []
        all_dones = []
        all_vals = []
        all_logps = []

        for i in range(len(agent_ids)):
            if len(obs_list[i]) == 0:
                continue
            all_obs.extend(obs_list[i])
            all_acts.extend(act_list[i])
            all_rews.extend(rew_list[i])
            all_dones.extend(done_list[i])
            all_vals.extend(val_list[i])
            all_logps.extend(logp_list[i])

        if len(all_obs) < 2:
            continue

        # Compute returns & advantages
        all_obs = np.array(all_obs)
        all_acts = np.array(all_acts)
        all_rews = np.array(all_rews)
        all_dones = np.array(all_dones)
        all_vals = np.array(all_vals)
        all_logps = np.array(all_logps)

        returns = np.zeros_like(all_rews)
        advantages = np.zeros_like(all_rews)
        gae = 0.0
        for t in reversed(range(len(all_rews))):
            if all_dones[t]:
                gae = 0.0
            delta = all_rews[t] + config.train.gamma * (0 if all_dones[t] else all_vals[t + 1 if t + 1 < len(all_vals) else t]) - all_vals[t]
            gae = delta + config.train.gamma * config.train.gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + all_vals[t]

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Mini-batch updates
        n_samples = len(all_obs)
        indices = np.arange(n_samples)

        for _ in range(config.train.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, config.train.batch_size):
                end = min(start + config.train.batch_size, n_samples)
                batch_idx = indices[start:end]

                obs_b = torch.as_tensor(all_obs[batch_idx], device=device).float()
                act_b = torch.as_tensor(all_acts[batch_idx], device=device).float()
                ret_b = torch.as_tensor(returns[batch_idx], device=device).float()
                adv_b = torch.as_tensor(advantages[batch_idx], device=device).float()
                old_logp_b = torch.as_tensor(all_logps[batch_idx], device=device).float()

                dist = model.policy.get_distribution(obs_b)
                logp = dist.log_prob(act_b)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - config.train.clip_range, 1 + config.train.clip_range) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                values = model.policy.predict_values(obs_b).flatten()
                value_loss = nn.functional.mse_loss(values, ret_b)

                loss = policy_loss + config.train.vf_coef * value_loss - config.train.ent_coef * entropy

                model.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.policy.parameters(), config.train.max_grad_norm)
                model.policy.optimizer.step()

        if total_steps % 10000 == 0 or total_steps >= config.train.total_timesteps:
            ckpt_path = os.path.join(config.train.model_dir, f"ippo_step_{total_steps}")
            model.save(ckpt_path)
            print(f"  [Checkpoint] Saved to {ckpt_path}")

    model.save(os.path.join(config.train.model_dir, "ippo_final"))
    env.close()
    print("[Independent PPO] Training complete.")


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train multi-UAV delivery agents")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "ippo"],
                        help="ppo=centralized | ippo=independent (param sharing)")
    parser.add_argument("--num_drones", type=int, default=4)
    parser.add_argument("--num_targets", type=int, default=4)
    parser.add_argument("--num_obstacles", type=int, default=10)
    parser.add_argument("--task_mode", type=str, default="reach",
                        choices=["reach", "delivery"])
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--n_envs", type=int, default=4, help="(ppo only) Num parallel envs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = FullConfig()
    config.env.num_drones = args.num_drones
    config.env.num_targets = args.num_targets
    config.env.num_obstacles = args.num_obstacles
    config.env.task_mode = args.task_mode
    config.train.algorithm = args.algo
    config.train.total_timesteps = args.total_steps
    config.train.n_envs = args.n_envs
    config.train.seed = args.seed
    config.train.device = args.device

    # Adjust PPO n_steps for short total steps
    if config.train.total_timesteps < 10000:
        config.train.n_steps = min(256, config.train.total_timesteps // config.train.n_envs)

    if args.algo == "ppo":
        train_centralized_ppo(config)
    elif args.algo == "ippo":
        train_independent_ppo(config)


if __name__ == "__main__":
    main()
