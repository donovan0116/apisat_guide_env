"""Diagnostic script: compare default vs tuned reward config for random policy."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs import QuadrotorDeliveryEnv
from utils.config import FullConfig
from core.reward import RewardConfig


def run_episodes(env, reward_config, num_episodes=20, max_steps=500, seed=42):
    totals = {
        "target_reached": [],
        "step_penalty": [],
        "collision": [],
        "energy": [],
        "shaping": [],
        "total": [],
    }
    lengths = []
    reached_counts = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        step = 0
        ep_components = {k: 0.0 for k in totals}
        while step < max_steps and not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            components = info.get("reward_components", {})
            for k in ep_components:
                if k == "total":
                    ep_components[k] += float(np.sum(reward))
                else:
                    ep_components[k] += float(np.sum(components.get(k, 0.0)))
            step += 1
            done = bool(np.any(terminated)) or bool(np.any(truncated))

        lengths.append(step)
        reached_counts.append(int(np.sum(env._target_assigned)))
        for k in totals:
            totals[k].append(ep_components[k])

    return {
        "lengths": lengths,
        "reached_counts": reached_counts,
        "totals": totals,
    }


def make_env(cfg, reward_config, max_steps, seed=42):
    cfg.reward = reward_config
    cfg.term.max_steps = max_steps
    cfg.env.max_steps = max_steps
    return QuadrotorDeliveryEnv(
        num_drones=cfg.env.num_drones,
        num_targets=cfg.env.num_targets,
        num_obstacles=cfg.env.num_obstacles,
        task_mode=cfg.env.task_mode,
        bounds=cfg.env.bounds_array,
        quad_params=cfg.quad,
        obs_config=cfg.obs,
        act_config=cfg.act,
        reward_config=reward_config,
        term_config=cfg.term,
        seed=seed,
    )


def print_results(label, results, num_targets):
    print(f"\n{label}")
    print(f"  Episode lengths: mean={np.mean(results['lengths']):.1f}, std={np.std(results['lengths']):.1f}")
    print(f"  Targets reached: mean={np.mean(results['reached_counts']):.2f}/{num_targets}")
    print(f"  Reward totals per episode (mean ± std):")
    for k, v in results["totals"].items():
        print(f"    {k:15s}: {np.mean(v):8.2f} ± {np.std(v):8.2f}")


if __name__ == "__main__":
    cfg = FullConfig()
    cfg.env.num_drones = 4
    cfg.env.num_targets = 4
    cfg.env.num_obstacles = 10
    cfg.env.task_mode = "reach"

    # Default reward
    default_reward = RewardConfig()
    env_default = make_env(cfg, default_reward, max_steps=500)
    results_default = run_episodes(env_default, default_reward, num_episodes=20)
    print_results("DEFAULT REWARD (max_steps=500)", results_default, cfg.env.num_targets)
    env_default.close()

    # Tuned reward
    tuned_reward = RewardConfig(
        target_reached=200.0,
        completion_bonus=500.0,
        distance_scale=0.005,
        step_penalty=0.1,
        obstacle_collision=-50.0,
        drone_collision=-30.0,
        out_of_bounds_penalty=-10.0,
        energy_coeff=0.001,
        use_shaping=True,
    )
    env_tuned = make_env(cfg, tuned_reward, max_steps=500)
    results_tuned = run_episodes(env_tuned, tuned_reward, num_episodes=20)
    print_results("TUNED REWARD (max_steps=500)", results_tuned, cfg.env.num_targets)
    env_tuned.close()

    # Tuned reward + longer episode
    env_tuned_long = make_env(cfg, tuned_reward, max_steps=2000)
    results_tuned_long = run_episodes(env_tuned_long, tuned_reward, num_episodes=20)
    print_results("TUNED REWARD (max_steps=2000)", results_tuned_long, cfg.env.num_targets)
    env_tuned_long.close()
