"""Diagnostic script: run a few episodes with a random policy and print reward components."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs import QuadrotorDeliveryEnv
from utils.config import FullConfig


def diagnose(num_episodes=10, num_drones=4, num_targets=4, num_obstacles=10, task_mode="reach", max_steps=500):
    cfg = FullConfig()
    cfg.env.num_drones = num_drones
    cfg.env.num_targets = num_targets
    cfg.env.num_obstacles = num_obstacles
    cfg.env.task_mode = task_mode
    cfg.env.max_steps = max_steps
    cfg.term.max_steps = max_steps

    env = QuadrotorDeliveryEnv(
        num_drones=num_drones,
        num_targets=num_targets,
        num_obstacles=num_obstacles,
        task_mode=task_mode,
        bounds=cfg.env.bounds_array,
        quad_params=cfg.quad,
        obs_config=cfg.obs,
        act_config=cfg.act,
        reward_config=cfg.reward,
        term_config=cfg.term,
        seed=42,
    )

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
        obs, info = env.reset(seed=42 + ep)
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

    env.close()

    print(f"Configuration: drones={num_drones}, targets={num_targets}, obstacles={num_obstacles}, mode={task_mode}, max_steps={max_steps}")
    print(f"Episode lengths: mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")
    print(f"Targets reached: mean={np.mean(reached_counts):.2f}/{num_targets}")
    print("\nReward component totals per episode (mean ± std):")
    for k, v in totals.items():
        print(f"  {k:15s}: {np.mean(v):8.2f} ± {np.std(v):8.2f}")
    print(f"\nComponent share of average total reward:")
    avg_total = np.mean(totals["total"])
    for k in ["target_reached", "step_penalty", "collision", "energy", "shaping"]:
        share = np.mean(totals[k]) / (abs(avg_total) + 1e-8) * 100
        print(f"  {k:15s}: {share:6.2f}% of total magnitude")


if __name__ == "__main__":
    print("=" * 70)
    print("DEFAULT CONFIG")
    print("=" * 70)
    diagnose()

    print("\n" + "=" * 70)
    print("LONGER EPISODE (2000 steps)")
    print("=" * 70)
    diagnose(max_steps=2000)
