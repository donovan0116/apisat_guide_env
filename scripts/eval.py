"""
Evaluation and visualization script.

Usage:
    python scripts/eval.py --model_path models/ppo_final.zip --render
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config import FullConfig
from envs import QuadrotorDeliveryEnv


def run_episode(env, model=None, render: bool = False, max_steps: int = 500):
    """Run a single episode and return metrics."""
    obs, info = env.reset()
    total_reward = np.zeros(env.num_drones)
    done = False
    step = 0

    while not done and step < max_steps:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Random policy for testing
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if render:
            env.render()

        done = np.any(terminated) or np.any(truncated) or step >= max_steps

    return {
        "total_reward": total_reward,
        "steps": step,
        "targets_reached": int(np.sum(info.get("target_assigned", [0]))),
        "total_targets": env.num_targets,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model (optional)")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--num_drones", type=int, default=4)
    parser.add_argument("--num_targets", type=int, default=4)
    parser.add_argument("--num_obstacles", type=int, default=10)
    parser.add_argument("--task_mode", type=str, default="reach",
                        choices=["reach", "delivery"])
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    config = FullConfig()
    config.env.num_drones = args.num_drones
    config.env.num_targets = args.num_targets
    config.env.num_obstacles = args.num_obstacles
    config.env.task_mode = args.task_mode
    config.term.max_steps = args.max_steps

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
        render_mode="human" if args.render else None,
        seed=args.seed,
    )

    model = None
    if args.model_path:
        try:
            from stable_baselines3 import PPO
            model = PPO.load(args.model_path)
            print(f"Loaded model from {args.model_path}")
        except ImportError:
            print("Stable-Baselines3 not installed. Using random policy.")
        except Exception as e:
            print(f"Failed to load model: {e}")

    all_rewards = []
    all_successes = []

    for ep in range(args.num_episodes):
        result = run_episode(env, model, render=args.render,
                             max_steps=args.max_steps)
        all_rewards.append(result["total_reward"])
        all_successes.append(result["targets_reached"] / result["total_targets"])

        print(f"Episode {ep + 1}/{args.num_episodes}: "
              f"steps={result['steps']}, "
              f"rewards={np.sum(result['total_reward']):.1f}, "
              f"targets={result['targets_reached']}/{result['total_targets']}")

    print(f"\nAverage reward: {np.mean([np.sum(r) for r in all_rewards]):.1f} "
          f"± {np.std([np.sum(r) for r in all_rewards]):.1f}")
    print(f"Average success rate: {np.mean(all_successes) * 100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
