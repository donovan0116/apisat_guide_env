"""
2D top-down renderer (matplotlib, no 3D projection).

Used as a lightweight alternative to the 3D renderer when the user wants a
fast, unoccluded overhead view (e.g. for training diagnostics).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle


def render_top_down(
    env,
    figsize: tuple = (7.0, 7.0),
    show_trail: bool = True,
    trail_length: int = 200,
) -> np.ndarray:
    """Render the current env state as a 2D top-down view.

    Parameters
    ----------
    env : QuadrotorDeliveryEnv
        The environment to render. Must expose ``_states``, ``_targets``,
        ``_obstacles``, ``_target_assignment`` and ``_carry_status``.
    figsize : tuple
        Figure size in inches.
    show_trail : bool
        Whether to draw drone path trails.
    trail_length : int
        Maximum number of trail points kept per drone.

    Returns
    -------
    np.ndarray
        RGB image of shape ``(H, W, 3)`` and dtype ``uint8``.
    """
    from collections import deque

    bounds = np.asarray(env.bounds, dtype=float)
    x_lim, y_lim = bounds[0], bounds[1]

    # Maintain trails across calls (caching on the env instance)
    if not hasattr(env, "_topdown_trails") or \
            len(getattr(env, "_topdown_trails", [])) != env.num_drones:
        env._topdown_trails = [deque(maxlen=trail_length) for _ in range(env.num_drones)]
    trails = env._topdown_trails
    for i in range(env.num_drones):
        trails[i].append(env._states[i, :2].copy())

    fig: Figure = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Top-Down View ({env.task_mode})")
    ax.grid(True, alpha=0.3)

    drone_colors = plt.cm.rainbow(np.linspace(0, 1, max(env.num_drones, 1)))

    # Obstacles
    for obs in env._obstacles:
        if obs.shape in ("cylinder", "sphere"):
            ax.add_patch(Circle(
                (obs.position[0], obs.position[1]), obs.size[0],
                facecolor="gray", alpha=0.4, edgecolor="black",
            ))
        else:  # box
            ax.add_patch(Rectangle(
                (obs.position[0] - obs.size[0] / 2,
                 obs.position[1] - obs.size[1] / 2),
                obs.size[0], obs.size[1],
                facecolor="gray", alpha=0.4, edgecolor="black",
            ))

    # Targets
    target_pos = np.array([t.position for t in env._targets])
    target_colors = []
    for t in env._targets:
        if t.reached:
            target_colors.append("gray")
        elif t.target_type.value == 1:  # PICKUP
            target_colors.append("tab:green")
        elif t.target_type.value == 2:  # DELIVERY
            target_colors.append("tab:red")
        else:
            target_colors.append("tab:blue")
    if len(target_pos) > 0:
        ax.scatter(target_pos[:, 0], target_pos[:, 1], c=target_colors,
                   s=120, marker="o", edgecolors="black", zorder=3)

    # Trails
    if show_trail:
        for i, tr in enumerate(trails):
            if len(tr) >= 2:
                arr = np.array(tr)
                ax.plot(arr[:, 0], arr[:, 1], "-", color=drone_colors[i],
                        lw=1.0, alpha=0.5, zorder=2)

    # Assignment lines
    for i in range(env.num_drones):
        aid = env._target_assignment[i]
        if 0 <= aid < len(env._targets):
            tp = env._targets[aid].position[:2]
            sp = env._states[i, :2]
            ax.plot([sp[0], tp[0]], [sp[1], tp[1]], "--",
                    color=drone_colors[i], alpha=0.3, lw=0.7, zorder=1)

    # Drones
    drone_pos = env._states[:, :2]
    ax.scatter(drone_pos[:, 0], drone_pos[:, 1], c=drone_colors,
               s=110, marker="^", edgecolors="black", zorder=4)
    for i in range(env.num_drones):
        ax.text(drone_pos[i, 0], drone_pos[i, 1] + 0.8, f"D{i}",
                fontsize=9, ha="center", va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="black",
                          alpha=0.85, linewidth=0.5),
                zorder=6)
        if env._carry_status[i]:
            ax.text(drone_pos[i, 0], drone_pos[i, 1] - 1.2, "📦",
                    fontsize=10, ha="center", zorder=5)

    # Target labels
    for j, t in enumerate(env._targets):
        if t.reached:
            label = f"T{j} ✓"
            color = "gray"
        elif t.target_type.value == 1:  # PICKUP
            label = f"T{j} ⬆"
            color = "darkgreen"
        elif t.target_type.value == 2:  # DELIVERY
            label = f"T{j} ⬇"
            color = "darkred"
        else:
            label = f"T{j}"
            color = "navy"
        ax.text(t.position[0], t.position[1] + 2.0, label,
                fontsize=9, ha="center", va="bottom", fontweight="bold",
                color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor=color,
                          alpha=0.9, linewidth=0.7))

    # Status overlay
    info_text = (
        f"step: {getattr(env, '_steps', 0)}    "
        f"targets: {int(np.sum(env._target_assigned))}/{env.num_targets}    "
        f"drones: {env.num_drones}"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            family="monospace", fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="black", alpha=0.7))

    fig.canvas.draw()
    try:
        buf = np.asarray(fig.canvas.buffer_rgba())
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    return buf[:, :, :3].copy()
