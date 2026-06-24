"""
State space definitions for the multi-UAV delivery environment.

Observation Types:
    - DRONE_STATE: 12-dim [pos(3), vel(3), euler(3), omega(3)]
    - RELATIVE_DRONE: 3-dim per other drone (relative position)
    - TARGET_INFO: 4-dim per target [rel_x, rel_y, rel_z, assigned_flag]
    - OBSTACLE_INFO: 3-dim per obstacle (relative position + distance)
    - SELF_INFO: 1-dim [carrying_package]

Global observation (centralized critic): full env state
Local observation (decentralized actor): partial + relative
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ObsConfig:
    """Configuration for observation space."""
    include_self_state: bool = True
    include_relative_drones: bool = True
    include_targets: bool = True
    include_obstacles: bool = True
    include_carry_status: bool = True
    max_num_drones: int = 10
    max_num_targets: int = 10
    max_num_obstacles: int = 20
    sensing_range: float = 50.0
    normalize: bool = True
    pos_bound: float = 50.0
    vel_bound: float = 10.0
    ang_vel_bound: float = 5.0

    @property
    def self_state_dim(self) -> int:
        return 12 if self.include_self_state else 0

    @property
    def relative_drones_dim(self) -> int:
        return self.max_num_drones * 3 if self.include_relative_drones else 0

    @property
    def targets_dim(self) -> int:
        return self.max_num_targets * 4 if self.include_targets else 0

    @property
    def obstacles_dim(self) -> int:
        return self.max_num_obstacles * 3 if self.include_obstacles else 0

    @property
    def carry_dim(self) -> int:
        return 1 if self.include_carry_status else 0

    @property
    def total_dim(self) -> int:
        return (self.self_state_dim + self.relative_drones_dim +
                self.targets_dim + self.obstacles_dim + self.carry_dim)


def build_local_obs(
    agent_idx: int,
    all_states: np.ndarray,  # [num_drones, 12]
    target_positions: np.ndarray,  # [num_targets, 3]
    target_assigned: np.ndarray,  # [num_targets], bool
    obstacle_positions: np.ndarray,  # [num_obstacles, 3]
    obstacle_radii: np.ndarray,  # [num_obstacles]
    carry_status: np.ndarray,  # [num_drones], bool
    config: ObsConfig,
) -> np.ndarray:
    """
    Build the local observation vector for a single agent.

    Returns a fixed-size flat observation vector padded for the
    maximum possible number of entities.

    Shape: (total_dim,)
    """
    obs_parts = []
    self_state = all_states[agent_idx]
    num_drones = len(all_states)
    num_targets = len(target_positions)

    # --- Self state ---
    if config.include_self_state:
        obs_parts.append(self_state)

    # --- Relative drone info ---
    if config.include_relative_drones:
        drone_info = np.zeros((config.max_num_drones, 3))
        count = 0
        for j in range(num_drones):
            if j != agent_idx and count < config.max_num_drones:
                rel_pos = all_states[j, 0:3] - self_state[0:3]
                drone_info[count] = rel_pos
                count += 1
        obs_parts.append(drone_info.flatten())

    # --- Target info ---
    if config.include_targets:
        target_info = np.zeros((config.max_num_targets, 4))
        for j in range(min(num_targets, config.max_num_targets)):
            rel_pos = target_positions[j] - self_state[0:3]
            target_info[j, 0:3] = rel_pos
            target_info[j, 3] = 1.0 if target_assigned[j] else 0.0
        obs_parts.append(target_info.flatten())

    # --- Obstacle info (nearest within sensing range) ---
    if config.include_obstacles:
        obs_info = np.zeros((config.max_num_obstacles, 3))
        if len(obstacle_positions) > 0:
            rel_obs = obstacle_positions - self_state[0:3]
            dists = np.linalg.norm(rel_obs, axis=1) - obstacle_radii
            mask = dists < config.sensing_range
            valid_rel = rel_obs[mask]
            valid_dists = dists[mask]
            n_valid = min(len(valid_rel), config.max_num_obstacles)
            if n_valid > 0:
                sort_idx = np.argsort(valid_dists)[:n_valid]
                obs_info[:n_valid, :] = valid_rel[sort_idx]
        obs_parts.append(obs_info.flatten())

    # --- Carry status ---
    if config.include_carry_status:
        obs_parts.append(np.array([1.0 if carry_status[agent_idx] else 0.0]))

    obs = np.concatenate(obs_parts)

    # --- Normalization ---
    if config.normalize:
        obs = _normalize_obs(obs, config, obs_parts)

    return obs.astype(np.float32)


def build_global_obs(
    all_states: np.ndarray,
    target_positions: np.ndarray,
    target_assigned: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    carry_status: np.ndarray,
    config: ObsConfig,
) -> np.ndarray:
    """Flattened global state for centralized critic."""
    num_drones = len(all_states)
    num_targets = len(target_positions)
    num_obstacles = len(obstacle_positions)

    parts = [all_states.flatten(),
             target_positions.flatten(),
             target_assigned.astype(np.float32),
             obstacle_positions.flatten() if num_obstacles > 0 else np.zeros(0),
             obstacle_radii.flatten() if num_obstacles > 0 else np.zeros(0),
             carry_status.astype(np.float32)]
    return np.concatenate(parts).astype(np.float32)


def _normalize_obs(obs: np.ndarray, config: ObsConfig, parts: list) -> np.ndarray:
    """Normalize observation components to [-1, 1] range."""
    norm_obs = obs.copy()
    offset = 0
    scale_pos = 1.0 / config.pos_bound
    scale_vel = 1.0 / config.vel_bound
    scale_omega = 1.0 / config.ang_vel_bound

    # Self state: [pos(3), vel(3), euler(3), omega(3)]
    if config.include_self_state:
        norm_obs[offset:offset + 3] *= scale_pos
        offset += 3
        norm_obs[offset:offset + 3] *= scale_vel
        offset += 3
        # euler: already in [-pi, pi] -> divide by pi
        norm_obs[offset:offset + 3] /= np.pi
        offset += 3
        norm_obs[offset:offset + 3] *= scale_omega
        offset += 3

    # Relative drone positions: scale by pos_bound
    if config.include_relative_drones:
        n = config.max_num_drones * 3
        norm_obs[offset:offset + n] *= scale_pos
        offset += n

    # Target info: relative pos (scaled) + assigned (0/1)
    if config.include_targets:
        for _ in range(config.max_num_targets):
            norm_obs[offset:offset + 3] *= scale_pos
            offset += 4  # skip assigned bit (already 0/1)

    # Obstacle info: relative pos (scaled)
    if config.include_obstacles:
        n = config.max_num_obstacles * 3
        norm_obs[offset:offset + n] *= scale_pos
        offset += n

    # Carry status already 0/1
    return np.clip(norm_obs, -1.0, 1.0)


# ── Semantic graph builder ───────────────────────────────────────────────


def build_semantic_graph(
    states: np.ndarray,
    target_positions: np.ndarray,
    target_assignment: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    carry_status: np.ndarray = None,
    use_llm: bool = False,
    llm_model: str = "gpt-4o",
    llm_api_key: str = "",
) -> "SemanticGraph":
    """Build a semantic interaction graph from environment state.

    This is the bridge between the environment and the SemGAT-MARL framework.
    It wraps the heuristic (or LLM) classifier to produce a SemanticGraph
    that can be consumed by the GAT policy.

    Parameters
    ----------
    states : (n_agents, 12) drone state vectors
    target_positions : (n_targets, 3)
    target_assignment : (n_agents,) int — assigned target per agent (-1 = none)
    obstacle_positions : (n_obstacles, 3)
    obstacle_radii : (n_obstacles,)
    carry_status : (n_agents,) bool or None
    use_llm : if True, attempt LLM-based classification
    llm_model / llm_api_key : passed to LLM classifier if used

    Returns
    -------
    SemanticGraph
    """
    from core.semantic import make_semantic_classifier

    classifier = make_semantic_classifier(
        use_llm=use_llm,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
    )
    return classifier.classify(
        states=states,
        target_positions=target_positions,
        target_assignment=target_assignment,
        obstacle_positions=obstacle_positions,
        obstacle_radii=obstacle_radii,
        carry_status=carry_status,
    )
