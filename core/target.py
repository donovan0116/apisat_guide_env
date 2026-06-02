"""
Target and obstacle definitions for the delivery task environment.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum


class TargetType(Enum):
    """Type of delivery target."""
    REACH = 0        # Simply fly to the target point
    PICKUP = 1       # Pick up package at this location
    DELIVERY = 2     # Deliver package to this location


@dataclass
class Target:
    """
    A delivery target in the environment.

    For delivery mode (PICKUP + DELIVERY pairing):
        - Each delivery task consists of one PICKUP and one DELIVERY target,
          linked by `pair_id`.
        - A drone must first visit the PICKUP, then the paired DELIVERY.

    For simple mode (REACH only):
        - Each target is an independent destination.
    """
    position: np.ndarray        # (3,) world coordinates
    target_type: TargetType
    pair_id: int = -1           # links PICKUP and DELIVERY (same id)
    priority: float = 1.0       # task priority/weight (higher = more important)
    time_window: Tuple[float, float] = (0.0, np.inf)  # (earliest, latest) time of arrival
    required_drones: int = 1    # number of drones needed (currently always 1)
    reward: float = 10.0        # reward for completing this target
    radius: float = 2.0         # success radius (distance to count as reached)
    reached: bool = False       # whether target has been reached


@dataclass
class Obstacle:
    """A 3D obstacle in the environment."""
    position: np.ndarray    # (3,) center position
    shape: str = "cylinder"  # "cylinder" | "box" | "sphere"
    size: np.ndarray = None  # (3,) dimensions (radius, radius, height for cylinder)

    def __post_init__(self):
        if self.size is None:
            self.size = np.array([2.0, 2.0, 5.0])

    @property
    def radius(self) -> float:
        if self.shape == "cylinder":
            return self.size[0]
        elif self.shape == "sphere":
            return self.size[0]
        elif self.shape == "box":
            return np.linalg.norm(self.size[:2]) / 1.5
        return self.size[0]

    def distance_to(self, point: np.ndarray) -> float:
        """
        Signed distance from a point to the obstacle surface.
        Negative means inside / collision.
        """
        dx = point - self.position
        if self.shape == "cylinder":
            # Horizontal distance from cylinder center axis
            h_dist = np.linalg.norm(dx[:2]) - self.size[0]
            # Vertical distance from top/bottom
            v_dist = np.abs(dx[2]) - self.size[2] / 2.0
            if v_dist <= 0:
                return h_dist  # within vertical range, horizontal dist matters
            # Outside vertical range
            h_dist_clamped = max(h_dist, 0)
            return -np.sqrt(h_dist_clamped ** 2 + v_dist ** 2)
        elif self.shape == "sphere":
            return np.linalg.norm(dx) - self.size[0]
        elif self.shape == "box":
            q = np.abs(dx) - self.size / 2.0
            return np.linalg.norm(np.maximum(q, 0)) + min(np.max(q), 0)
        return float("inf")


def generate_random_obstacles(
    num_obstacles: int,
    bounds: np.ndarray,        # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    min_distance: float = 5.0,  # minimum distance between obstacles
    target_positions: np.ndarray = None,  # positions to avoid
    target_clearance: float = 5.0,
    rng: np.random.Generator = None,
) -> List[Obstacle]:
    """Generate random obstacles avoiding target positions."""
    if rng is None:
        rng = np.random.default_rng()

    obstacles = []
    max_attempts = num_obstacles * 100
    attempts = 0

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        pos = rng.uniform(bounds[:, 0], bounds[:, 1])
        shape = rng.choice(["cylinder", "sphere", "box"], p=[0.6, 0.2, 0.2])

        if shape == "cylinder":
            size = np.array([rng.uniform(1.0, 4.0), 0, rng.uniform(3.0, 10.0)])
        elif shape == "sphere":
            s = rng.uniform(1.0, 3.0)
            size = np.array([s, s, s])
        else:
            size = rng.uniform([1.0, 1.0, 2.0], [4.0, 4.0, 8.0])

        candidate = Obstacle(position=pos, shape=shape, size=size)

        # Check against existing obstacles
        valid = True
        for obs in obstacles:
            if np.linalg.norm(pos - obs.position) < min_distance:
                valid = False
                break

        # Check against target positions
        if valid and target_positions is not None:
            for tp in target_positions:
                if np.linalg.norm(pos - tp) < candidate.radius + target_clearance:
                    valid = False
                    break

        if valid:
            obstacles.append(candidate)

        attempts += 1

    return obstacles


def generate_random_targets(
    num_targets: int,
    bounds: np.ndarray,
    mode: str = "reach",           # "reach" | "delivery"
    drone_positions: np.ndarray = None,
    obstacle_positions: np.ndarray = None,
    obstacle_radii: np.ndarray = None,
    min_dist_drone: float = 10.0,
    min_dist_obstacle: float = 5.0,
    rng: np.random.Generator = None,
) -> List[Target]:
    """Generate random targets (simple reach or pickup+delivery pairs)."""
    if rng is None:
        rng = np.random.default_rng()

    targets = []
    max_attempts = num_targets * 100

    if mode == "delivery":
        # Generate pickup-delivery pairs
        num_pairs = num_targets // 2
        for pair_id in range(num_pairs):
            attempts = 0
            while attempts < max_attempts:
                pickup_pos = rng.uniform(bounds[:, 0], bounds[:, 1])
                delivery_pos = rng.uniform(bounds[:, 0], bounds[:, 1])

                # Ensure reasonable distance between pickup and delivery
                if np.linalg.norm(delivery_pos - pickup_pos) < 10.0:
                    attempts += 1
                    continue

                if not _check_position_valid(pickup_pos, drone_positions,
                                              obstacle_positions, obstacle_radii,
                                              min_dist_drone, min_dist_obstacle):
                    attempts += 1
                    continue
                if not _check_position_valid(delivery_pos, drone_positions,
                                              obstacle_positions, obstacle_radii,
                                              min_dist_drone, min_dist_obstacle):
                    attempts += 1
                    continue

                targets.append(Target(position=pickup_pos, target_type=TargetType.PICKUP,
                                      pair_id=pair_id))
                targets.append(Target(position=delivery_pos, target_type=TargetType.DELIVERY,
                                      pair_id=pair_id))
                break
    else:
        # Simple reach targets
        for _ in range(num_targets):
            attempts = 0
            while attempts < max_attempts:
                pos = rng.uniform(bounds[:, 0], bounds[:, 1])
                if _check_position_valid(pos, drone_positions,
                                          obstacle_positions, obstacle_radii,
                                          min_dist_drone, min_dist_obstacle):
                    targets.append(Target(position=pos, target_type=TargetType.REACH))
                    break
                attempts += 1

    return targets


def _check_position_valid(
    pos: np.ndarray,
    drone_positions: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    min_dist_drone: float,
    min_dist_obstacle: float,
) -> bool:
    if drone_positions is not None and len(drone_positions) > 0:
        dist_to_drones = np.linalg.norm(drone_positions[:, :2] - pos[:2], axis=1)
        if np.any(dist_to_drones < min_dist_drone):
            return False
    if obstacle_positions is not None and len(obstacle_positions) > 0:
        dist_to_obs = np.linalg.norm(obstacle_positions[:, :2] - pos[:2], axis=1)
        if np.any(dist_to_obs < (obstacle_radii + min_dist_obstacle)):
            return False
    return True
