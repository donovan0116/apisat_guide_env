"""
Evaluation and visualization script.

Usage:
    python scripts/eval.py --model_path models/ppo_final.zip --render
    python scripts/eval.py --render --render_mode human
    python scripts/eval.py --record --record_path logs/eval.mp4 --num_episodes 3
    python scripts/eval.py --record --record_path logs/eval.gif
    python scripts/eval.py --top_down
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config import FullConfig
from envs import QuadrotorDeliveryEnv


def _make_policy_fn(model):
    """Wrap a stable-baselines3 model as ``policy(obs) -> action``."""
    if model is None:
        return None

    def _policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return _policy


def run_episode(env, model=None, render: bool = False, render_mode: str = "human",
                max_steps: int = 500, top_down: bool = False):
    """Run a single episode and return metrics."""
    obs, info = env.reset()
    total_reward = np.zeros(env.num_drones)
    step = 0

    while step < max_steps:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if render:
            if top_down:
                env_top = env
                from utils.topdown import render_top_down
                frame = render_top_down(env_top)
                if frame is not None:
                    pass  # headless; nothing to display
            else:
                env.render()

        if bool(np.any(terminated)) or bool(np.any(truncated)):
            break

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
    parser.add_argument("--render", action="store_true",
                        help="Render the environment in real time")
    parser.add_argument("--render_mode", type=str, default="human",
                        choices=["human", "rgb_array", "rgb_array_list",
                                 "video", "top_down"],
                        help="Gymnasium-style render mode")
    parser.add_argument("--record", action="store_true",
                        help="Record each episode to a video file")
    parser.add_argument("--record_path", type=str, default="logs/eval.mp4",
                        help="Output video path (extension determines format)")
    parser.add_argument("--record_fps", type=int, default=30,
                        help="Recording FPS")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier for live rendering")
    parser.add_argument("--top_down", action="store_true",
                        help="Use the 2D top-down view (overrides --render_mode)")
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

    # Choose render mode
    if args.record:
        render_mode = "rgb_array"
    elif args.top_down:
        render_mode = "top_down"
    elif args.render:
        render_mode = args.render_mode
    else:
        render_mode = None

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
        render_mode=render_mode,
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
        if args.record:
            # Recording path: run a clean episode using the high-level API
            from utils.visualization import record_episode
            t0 = time.time()
            policy = _make_policy_fn(model)
            path = record_episode(
                env, args.record_path,
                fps=args.record_fps,
                max_steps=args.max_steps,
                policy=policy,
            )
            # Compute metrics from the recorded env state
            metrics = run_episode(env, model, render=False, max_steps=args.max_steps)
            elapsed = time.time() - t0
            print(f"[record] wrote {path} in {elapsed:.1f}s")
        else:
            metrics = run_episode(
                env, model,
                render=args.render or args.top_down,
                render_mode=render_mode or "human",
                max_steps=args.max_steps,
                top_down=args.top_down,
            )

        all_rewards.append(metrics["total_reward"])
        all_successes.append(
            metrics["targets_reached"] / max(metrics["total_targets"], 1)
        )

        print(f"Episode {ep + 1}/{args.num_episodes}: "
              f"steps={metrics['steps']}, "
              f"rewards={np.sum(metrics['total_reward']):.1f}, "
              f"targets={metrics['targets_reached']}/{metrics['total_targets']}")

    print(f"\nAverage reward: {np.mean([np.sum(r) for r in all_rewards]):.1f} "
          f"± {np.std([np.sum(r) for r in all_rewards]):.1f}")
    print(f"Average success rate: {np.mean(all_successes) * 100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
