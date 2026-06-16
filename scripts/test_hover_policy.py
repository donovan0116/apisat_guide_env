"""Test near-hover policy survival and reward."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs import QuadrotorDeliveryEnv
from utils.config import FullConfig


def run_near_hover(num_episodes=10, max_steps=2000, action_noise=0.1, seed=42):
    cfg = FullConfig()
    env = QuadrotorDeliveryEnv(
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
        seed=seed,
    )

    total_rewards = []
    lengths = []
    reached = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        step = 0
        done = False
        while step < max_steps and not done:
            # Near-hover action: thrust slightly above hover, small random torques
            action = np.zeros((env.num_drones, env.action_space.shape[1]), dtype=np.float32)
            action[:, 0] = -0.1  # normalized thrust near hover
            action[:, 1:] = np.random.normal(0, action_noise, size=(env.num_drones, 3))
            action = np.clip(action, -1, 1)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(np.sum(reward))
            step += 1
            done = bool(np.any(terminated)) or bool(np.any(truncated))
        total_rewards.append(total_reward)
        lengths.append(step)
        reached.append(int(np.sum(env._target_assigned)))

    env.close()
    print(f"Near-hover policy (noise={action_noise}):")
    print(f"  Episode lengths: mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")
    print(f"  Targets reached: mean={np.mean(reached):.2f}/{cfg.env.num_targets}")
    print(f"  Total reward: mean={np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")


if __name__ == "__main__":
    for noise in [0.0, 0.05, 0.1, 0.2, 0.5]:
        run_near_hover(action_noise=noise)
        print()
