"""
Action space definitions with normalization and clipping.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ActionConfig:
    """Configuration for action space."""
    thrust_range: tuple = (0.0, 0.6)  # N
    torque_range: tuple = (-0.05, 0.05)  # N*m
    use_normalized_actions: bool = True   # If True, actions in [-1, +1]
    action_noise_std: float = 0.0


def normalize_action(raw_action: np.ndarray, config: ActionConfig) -> np.ndarray:
    """
    Convert normalized action [-1,1]^4 to physical units [thrust, taux, tauy, tauz].

    Args:
        raw_action: (4,) array in [-1, -1]^4
            raw_action[0]: thrust  (-1 = min, +1 = max)
            raw_action[1:]: torques (-1 = min, +1 = max)

    Returns:
        physical_action: (4,) array [thrust(N), taux(N*m), tauy(N*m), tauz(N*m)]
    """
    if not config.use_normalized_actions:
        return raw_action

    thrust = _map_range(raw_action[0],
                        -1.0, 1.0,
                        config.thrust_range[0], config.thrust_range[1])
    taux = _map_range(raw_action[1],
                      -1.0, 1.0,
                      config.torque_range[0], config.torque_range[1])
    tauy = _map_range(raw_action[2],
                      -1.0, 1.0,
                      config.torque_range[0], config.torque_range[1])
    tauz = _map_range(raw_action[3],
                      -1.0, 1.0,
                      config.torque_range[0], config.torque_range[1])

    return np.array([thrust, taux, tauy, tauz])


def _map_range(x, in_min, in_max, out_min, out_max):
    return out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)
