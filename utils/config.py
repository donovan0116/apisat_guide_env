"""
Configuration system for the environment and training.

Uses dataclasses for type-safe configuration with default values.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np

from core.dynamics import QuadrotorParams
from core.state import ObsConfig
from core.action import ActionConfig
from core.reward import RewardConfig
from core.termination import TerminationConfig


@dataclass
class EnvConfig:
    """Environment configuration."""

    num_drones: int = 4
    num_targets: int = 4
    num_obstacles: int = 10
    task_mode: str = "reach"  # "reach" | "delivery"
    bounds: Tuple = (-50, 50, -50, 50, 0, 30)  # (xmin, xmax, ymin, ymax, zmin, zmax)
    aggregate_phy_steps: int = 1
    max_steps: int = 2000

    @property
    def bounds_array(self) -> np.ndarray:
        b = self.bounds
        return np.array([[b[0], b[1]], [b[2], b[3]], [b[4], b[5]]])


@dataclass
class TrainingConfig:
    """Training configuration."""

    algorithm: str = "ppo"  # "ppo" | "sac" | "mappo" | "maddpg"
    total_timesteps: int = 1_000_000
    n_envs: int = 4
    seed: int = 42
    device: str = "cpu"
    log_dir: str = "./logs"
    model_dir: str = "./models"
    eval_freq: int = 10_000
    n_eval_episodes: int = 10

    # PPO-specific
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network
    policy_hidden_size: int = 256
    policy_n_layers: int = 3
    value_hidden_size: int = 256
    value_n_layers: int = 3


@dataclass
class FullConfig:
    """Top-level configuration aggregating all sub-configs."""

    env: EnvConfig = field(default_factory=EnvConfig)
    quad: QuadrotorParams = field(default_factory=QuadrotorParams)
    obs: ObsConfig = field(default_factory=ObsConfig)
    act: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    term: TerminationConfig = field(default_factory=TerminationConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        self.obs.max_num_drones = self.env.num_drones
        self.obs.max_num_targets = self.env.num_targets
        self.obs.max_num_obstacles = self.env.num_obstacles
        self.term.max_steps = self.env.max_steps


def get_default_config() -> FullConfig:
    """Return default configuration."""
    return FullConfig()
