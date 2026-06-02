"""
Reward function for the multi-UAV delivery task assignment environment.

Reward Components:
    1. target_reached:     positive reward when a drone reaches a target
    2. step_penalty:       small negative reward per time step (encourages efficiency)
    3. collision_penalty:  large negative reward for obstacle/drone-drone collisions
    4. energy_penalty:     penalty proportional to control effort
    5. completion_bonus:   bonus when all targets are completed
    6. hover_bonus:        small bonus for staying near grounded state (optional)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RewardConfig:
    """Reward function configuration."""
    target_reached: float = 20.0
    step_penalty: float = 0.01
    obstacle_collision: float = -50.0
    drone_collision: float = -30.0
    out_of_bounds_penalty: float = -10.0
    energy_coeff: float = 0.001
    completion_bonus: float = 100.0
    distance_scale: float = 0.1      # shaping reward: scale for distance reduction
    use_shaping: bool = True
    collision_radius: float = 1.0    # drone collision radius
    ground_penalty: bool = True      # penalty for flying too low (z < 0)


class RewardCalculator:
    """Computes per-agent and global rewards."""

    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()

    def compute(
        self,
        states: np.ndarray,              # [n_drones, 12] drone states
        target_positions: np.ndarray,    # [n_targets, 3]
        target_assigned: np.ndarray,     # [n_targets] bool
        target_assignment: np.ndarray,   # [n_drones] int, -1 = unassigned
        obstacle_positions: np.ndarray,  # [n_obstacles, 3]
        obstacle_radii: np.ndarray,      # [n_obstacles]
        actions: np.ndarray,             # [n_drones, 4] physical actions
        bounds: np.ndarray,              # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
        all_done: bool,
    ) -> Dict[str, np.ndarray]:
        """
        Compute rewards for all agents.

        Returns:
            Dict with:
                "agent_rewards":  (n_drones,) per-agent reward
                "team_reward":    scalar team reward
                "components":     dict of component breakdowns
        """
        n_drones = len(states)
        config = self.config

        agent_rewards = np.zeros(n_drones)
        components = {
            "target_reached": np.zeros(n_drones),
            "step_penalty": np.zeros(n_drones),
            "collision": np.zeros(n_drones),
            "energy": np.zeros(n_drones),
            "shaping": np.zeros(n_drones),
        }

        # === Step penalty (all agents) ===
        components["step_penalty"][:] = -config.step_penalty

        # === Target reached reward ===
        for i in range(n_drones):
            aid = target_assignment[i]
            if aid >= 0 and aid < len(target_positions):
                dist = np.linalg.norm(states[i, :3] - target_positions[aid])
                if dist < 2.0:  # success radius (TODO: parameterize)
                    components["target_reached"][i] = config.target_reached

        # === Distance shaping reward ===
        if config.use_shaping:
            for i in range(n_drones):
                aid = target_assignment[i]
                if aid >= 0 and aid < len(target_positions):
                    dist = np.linalg.norm(states[i, :3] - target_positions[aid])
                    components["shaping"][i] = -config.distance_scale * dist

        # === Collision penalties ===
        # Obstacle collisions
        for i in range(n_drones):
            pos_i = states[i, :3]
            for j, obs_pos in enumerate(obstacle_positions):
                dist_to_center = np.linalg.norm(pos_i[:2] - obs_pos[:2])
                if dist_to_center < obstacle_radii[j] + config.collision_radius:
                    components["collision"][i] = config.obstacle_collision
                    break

        # Drone-drone collisions
        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                dist = np.linalg.norm(states[i, :3] - states[j, :3])
                if dist < 2 * config.collision_radius:
                    # Penalize both
                    if components["collision"][i] > -1e6:
                        components["collision"][i] = config.drone_collision
                    if components["collision"][j] > -1e6:
                        components["collision"][j] = config.drone_collision

        # Out of bounds
        for i in range(n_drones):
            pos = states[i, :3]
            if (pos[0] < bounds[0, 0] or pos[0] > bounds[0, 1] or
                pos[1] < bounds[1, 0] or pos[1] > bounds[1, 1] or
                pos[2] < bounds[2, 0] or pos[2] > bounds[2, 1]):
                components["collision"][i] += config.out_of_bounds_penalty

        # Ground penalty
        if config.ground_penalty:
            for i in range(n_drones):
                if states[i, 2] < 0.0:
                    components["collision"][i] += config.out_of_bounds_penalty

        # === Energy penalty ===
        for i in range(n_drones):
            thrust = actions[i, 0]
            torque_mag = np.sum(np.abs(actions[i, 1:4]))
            components["energy"][i] = -config.energy_coeff * (thrust + torque_mag)

        # === Sum agent rewards ===
        for key in components:
            agent_rewards += components[key]

        # === Team reward (completion bonus) ===
        team_reward = 0.0
        if np.all(target_assigned) and all_done:
            team_reward = config.completion_bonus
            agent_rewards += team_reward / n_drones

        return {
            "agent_rewards": agent_rewards,
            "team_reward": team_reward,
            "components": components,
        }
