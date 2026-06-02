"""
Termination conditions for the environment.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TerminationConfig:
    """Configuration for termination conditions."""
    max_steps: int = 500
    target_radius: float = 2.0
    collision_radius: float = 1.0
    out_of_bounds_kill: bool = True
    terminate_on_collision: bool = True
    terminate_on_all_done: bool = True


class TerminationChecker:
    """Checks termination conditions and computes `done` flags."""

    def __init__(self, config: TerminationConfig = None):
        self.config = config or TerminationConfig()
        self._step_count = 0

    def reset(self):
        self._step_count = 0

    def check(
        self,
        states: np.ndarray,              # [n_drones, 12]
        target_positions: np.ndarray,    # [n_targets, 3]
        target_assigned: np.ndarray,     # [n_targets] bool
        target_assignment: np.ndarray,   # [n_drones] int
        obstacle_positions: np.ndarray,  # [n_obstacles, 3]
        obstacle_radii: np.ndarray,      # [n_obstacles]
        carry_status: np.ndarray,        # [n_drones] bool
        bounds: np.ndarray,              # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
        task_type: str = "reach",
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Returns:
            terminated: (n_drones,) per-agent termination
            truncated: (n_drones,) per-agent truncation (max steps)
            all_done: bool, global done flag
        """
        self._step_count += 1
        n_drones = len(states)
        config = self.config

        terminated = np.zeros(n_drones, dtype=bool)
        truncated = np.zeros(n_drones, dtype=bool)

        # --- Max steps ---
        if self._step_count >= config.max_steps:
            truncated[:] = True
            return terminated, truncated, True

        # --- Target completion ---
        for i in range(n_drones):
            aid = target_assignment[i]
            if aid >= 0 and aid < len(target_positions):
                dist = np.linalg.norm(states[i, :3] - target_positions[aid])
                if dist < config.target_radius:
                    if task_type == "delivery":
                        # In delivery mode, drone needs to carry package
                        if carry_status[i]:
                            # Delivery completed for this drone
                            terminated[i] = True
                    else:
                        # In reach mode, simply reaching is enough
                        terminated[i] = True

        # --- Collisions ---
        if config.terminate_on_collision:
            # Obstacle collisions
            for i in range(n_drones):
                if terminated[i]:
                    continue
                pos = states[i, :3]
                for j, obs_pos in enumerate(obstacle_positions):
                    h_dist = np.linalg.norm(pos[:2] - obs_pos[:2])
                    v_dist = abs(pos[2] - obs_pos[2])
                    # Simplified cylinder check
                    if h_dist < obstacle_radii[j] + config.collision_radius:
                        if v_dist < 5.0:  # HACK: assume height=10, center z at 0
                            terminated[i] = True
                            break

            # Drone-drone collisions
            for i in range(n_drones):
                if terminated[i]:
                    continue
                for j in range(i + 1, n_drones):
                    if terminated[j]:
                        continue
                    dist = np.linalg.norm(states[i, :3] - states[j, :3])
                    if dist < 2 * config.collision_radius:
                        terminated[i] = True
                        terminated[j] = True

        # --- Out of bounds ---
        if config.out_of_bounds_kill:
            for i in range(n_drones):
                if terminated[i]:
                    continue
                p = states[i, :3]
                if (p[0] < bounds[0, 0] or p[0] > bounds[0, 1] or
                    p[1] < bounds[1, 0] or p[1] > bounds[1, 1] or
                    p[2] < bounds[2, 0] or p[2] > bounds[2, 1]):
                    terminated[i] = True

        # --- All done ---
        all_done = np.all(terminated)
        if config.terminate_on_all_done and all_done:
            return terminated, truncated, True

        return terminated, truncated, all_done
