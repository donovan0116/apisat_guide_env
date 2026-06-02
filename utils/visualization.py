"""
High-level visualization API for the multi-UAV delivery environment.

Three primary entry points:

* :func:`play_episode` - offline animated replay (great for Jupyter / scripted
  inspection of a finished episode). Produces an HTML5 video or animated GIF
  via :class:`matplotlib.animation.FuncAnimation`.
* :func:`record_episode` - records an episode to a video file (mp4/gif).
* :func:`live_render` - opens an interactive matplotlib window that displays
  the environment in real time, with a fixed wall-clock FPS.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from envs.quadrotor_delivery_v0 import QuadrotorDeliveryEnv
from utils.rendering import (
    Matplotlib3DRenderer,
    RenderEvent,
    RenderState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_targets_from_state(env: QuadrotorDeliveryEnv) -> List[Any]:
    return env.targets


def _build_state_from_env(
    env: QuadrotorDeliveryEnv,
    show_trail: bool,
) -> RenderState:
    """Wrap the current env state into a :class:`RenderState`."""
    info = {
        "steps": getattr(env, "_steps", 0),
        "reward": getattr(env, "_last_rewards", None),
        "completed": int(np.sum(env._target_assigned)),
        "total_targets": env.num_targets,
        "mode": env.task_mode,
    }
    return RenderState(
        states=env._states.copy(),
        targets=_detect_targets_from_state(env),
        obstacles=env._obstacles,
        assignment=env._target_assignment.copy(),
        carry_status=env._carry_status.copy(),
        info=info,
    )


def _infer_action_from_policy(
    env: QuadrotorDeliveryEnv,
    policy: Optional[Callable[[Dict[str, np.ndarray]], np.ndarray]],
    obs: Dict[str, np.ndarray],
) -> np.ndarray:
    if policy is None:
        return env.action_space.sample()
    action = policy(obs)
    action = np.asarray(action, dtype=np.float32)
    if action.shape != env.action_space.shape:
        action = np.broadcast_to(action, env.action_space.shape).copy()
    return np.clip(action, env.action_space.low, env.action_space.high)


# ---------------------------------------------------------------------------
# record_episode
# ---------------------------------------------------------------------------


def record_episode(
    env: QuadrotorDeliveryEnv,
    save_path: str,
    *,
    fps: int = 30,
    max_steps: int = 500,
    policy: Optional[Callable] = None,
    codec: str = "libx264",
    dpi: int = 100,
    show_trail: bool = True,
    show_top_down: bool = True,
) -> str:
    """Record a single episode to a video file.

    Parameters
    ----------
    env : QuadrotorDeliveryEnv
        Must be created with ``render_mode="rgb_array"`` (or this function
        will set it implicitly).
    save_path : str
        Output file path. Format is inferred from extension
        (``.mp4``, ``.gif``, ``.avi``, ...).
    fps : int
        Target frames per second.
    max_steps : int
        Maximum env steps per episode.
    policy : callable, optional
        ``policy(obs) -> action`` where ``action`` has shape
        ``(num_drones, 4)``. ``None`` selects a random policy.
    codec : str
        Codec passed to :class:`FFMpegWriter` when writing mp4.
    dpi : int
        Output resolution in DPI.

    Returns
    -------
    str
        The path the video was written to.
    """
    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    ext = os.path.splitext(save_path)[1].lower()

    # Force rgb_array mode for video capture
    if env.render_mode != "rgb_array":
        env.render_mode = "rgb_array"

    renderer = Matplotlib3DRenderer(
        env.bounds,
        show_trail=show_trail,
        show_top_down=show_top_down,
    )
    try:
        obs, info = env.reset()
        renderer.update(_build_state_from_env(env, show_trail))
        frame = renderer.get_frame()
        h, w = frame.shape[:2]

        if ext in (".mp4", ".mov", ".avi", ".mkv"):
            try:
                writer = FFMpegWriter(fps=fps, codec=codec, bitrate=1800)
            except (RuntimeError, ValueError) as e:
                print(f"[record_episode] FFMpegWriter unavailable ({e}); "
                      "falling back to GIF.")
                save_path = os.path.splitext(save_path)[0] + ".gif"
                writer = PillowWriter(fps=fps)
        elif ext == ".gif":
            writer = PillowWriter(fps=fps)
        else:
            raise ValueError(f"Unsupported video extension: {ext}")

        step = 0
        with writer.saving(renderer._fig, save_path, dpi=dpi):
            while step < max_steps:
                action = _infer_action_from_policy(env, policy, obs)
                obs, _reward, terminated, truncated, _info = env.step(action)
                step += 1
                renderer.update(_build_state_from_env(env, show_trail))
                writer.grab_frame()
                if bool(np.any(terminated)) or bool(np.any(truncated)):
                    break
        return save_path
    finally:
        renderer.close()


# ---------------------------------------------------------------------------
# play_episode (FuncAnimation)
# ---------------------------------------------------------------------------


def play_episode(
    env: QuadrotorDeliveryEnv,
    *,
    fps: int = 30,
    max_steps: int = 500,
    policy: Optional[Callable] = None,
    save_path: Optional[str] = None,
    show_trail: bool = True,
    show_top_down: bool = True,
    dpi: int = 100,
    codec: str = "libx264",
) -> Dict[str, Any]:
    """Run an episode and play it back as a smooth animation.

    Parameters
    ----------
    env : QuadrotorDeliveryEnv
        Environment to simulate.
    fps : int
        Target playback FPS.
    max_steps : int
        Maximum env steps per episode.
    policy : callable, optional
        ``policy(obs) -> action``; ``None`` -> random.
    save_path : str, optional
        If given, save the animation to this file (mp4/gif).
    show_trail, show_top_down : bool
        Renderer flags.
    dpi : int
        Resolution used when saving.
    codec : str
        Codec used for mp4 saving.

    Returns
    -------
    dict
        ``{"frames": int, "duration_sec": float, "save_path": Optional[str]}``
    """
    # Pre-roll the episode and record all states
    env.reset()
    states_history: List[np.ndarray] = [env._states.copy()]
    assignment_history: List[np.ndarray] = [env._target_assignment.copy()]
    carry_history: List[np.ndarray] = [env._carry_status.copy()]
    target_status: List[List[bool]] = [[t.reached for t in env.targets]]
    rewards_history: List[np.ndarray] = []
    completed_history: List[int] = [int(np.sum(env._target_assigned))]

    obs, _ = env.reset()
    step = 0
    while step < max_steps:
        action = _infer_action_from_policy(env, policy, obs)
        obs, reward, terminated, truncated, _info = env.step(action)
        step += 1
        states_history.append(env._states.copy())
        assignment_history.append(env._target_assignment.copy())
        carry_history.append(env._carry_status.copy())
        target_status.append([t.reached for t in env.targets])
        rewards_history.append(reward)
        completed_history.append(int(np.sum(env._target_assigned)))
        if bool(np.any(terminated)) or bool(np.any(truncated)):
            break

    n_frames = len(states_history)
    duration = n_frames / float(fps)

    # Build a renderer on a fresh figure
    renderer = Matplotlib3DRenderer(
        env.bounds,
        show_trail=show_trail,
        show_top_down=show_top_down,
    )

    # Apply the first state
    targets_snapshot = [t for t in env.targets]
    renderer.update(RenderState(
        states=states_history[0],
        targets=targets_snapshot,
        obstacles=env._obstacles,
        assignment=assignment_history[0],
        carry_status=carry_history[0],
        info={"steps": 0, "completed": completed_history[0],
              "total_targets": env.num_targets, "mode": env.task_mode},
    ))

    def _update(frame_idx: int):
        for j, reached in enumerate(target_status[frame_idx]):
            targets_snapshot[j].reached = reached
        info = {
            "steps": frame_idx,
            "completed": completed_history[frame_idx],
            "total_targets": env.num_targets,
            "mode": env.task_mode,
        }
        if frame_idx > 0:
            info["reward"] = rewards_history[frame_idx - 1]
        renderer.update(RenderState(
            states=states_history[frame_idx],
            targets=targets_snapshot,
            obstacles=env._obstacles,
            assignment=assignment_history[frame_idx],
            carry_status=carry_history[frame_idx],
            info=info,
        ))
        return []

    try:
        anim = FuncAnimation(
            renderer._fig,
            _update,
            frames=n_frames,
            interval=1000.0 / fps,
            blit=False,
            repeat=True,
        )

        if save_path is not None:
            save_path = os.path.abspath(save_path)
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            ext = os.path.splitext(save_path)[1].lower()
            if ext in (".mp4", ".mov", ".avi", ".mkv"):
                try:
                    writer = FFMpegWriter(fps=fps, codec=codec, bitrate=1800)
                except (RuntimeError, ValueError):
                    save_path = os.path.splitext(save_path)[0] + ".gif"
                    writer = PillowWriter(fps=fps)
            elif ext == ".gif":
                writer = PillowWriter(fps=fps)
            else:
                raise ValueError(f"Unsupported video extension: {ext}")
            anim.save(save_path, writer=writer, dpi=dpi)

        return {
            "frames": n_frames,
            "duration_sec": duration,
            "save_path": save_path,
        }
    finally:
        # FuncAnimation keeps a reference; explicitly close if no save
        if save_path is None:
            try:
                plt.show(block=False)
                plt.pause(min(0.1, duration))
            except Exception:
                pass
        renderer.close()


# ---------------------------------------------------------------------------
# live_render
# ---------------------------------------------------------------------------


def live_render(
    env: QuadrotorDeliveryEnv,
    *,
    policy: Optional[Callable] = None,
    fps: int = 30,
    max_steps: int = 500,
    show_trail: bool = True,
    show_top_down: bool = True,
) -> Dict[str, Any]:
    """Open an interactive matplotlib window and step the env in real time.

    The window is non-blocking; the function returns after the last step or
    immediately if the env is closed.

    Parameters
    ----------
    env : QuadrotorDeliveryEnv
        Must be created with ``render_mode="human"`` (or ``None``; we will
        delegate to ``env.render()``).
    """
    plt.ion()
    renderer = Matplotlib3DRenderer(
        env.bounds,
        show_trail=show_trail,
        show_top_down=show_top_down,
    )
    target_dt = 1.0 / float(fps)
    obs, _ = env.reset()
    renderer.update(_build_state_from_env(env, show_trail))
    plt.show(block=False)

    step = 0
    t_start = time.monotonic()
    try:
        while step < max_steps:
            t0 = time.monotonic()
            action = _infer_action_from_policy(env, policy, obs)
            obs, _reward, terminated, truncated, _info = env.step(action)
            step += 1
            renderer.update(_build_state_from_env(env, show_trail))
            plt.pause(max(0.0, target_dt - (time.monotonic() - t0)))
            if bool(np.any(terminated)) or bool(np.any(truncated)):
                break
    finally:
        elapsed = time.monotonic() - t_start
        renderer.close()
        plt.ioff()
    return {"steps": step, "elapsed_sec": elapsed, "fps_actual": step / max(elapsed, 1e-9)}
