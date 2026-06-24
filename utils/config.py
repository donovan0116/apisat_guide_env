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
    aggregate_phy_steps: int = 20
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
    gamma: float = 0.995
    gae_lambda: float = 0.97
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
class VAEConfig:
    """Dual-Head Variational Autoencoder configuration.

    The DH-VAE learns a latent representation z that jointly encodes
    physical dynamics (reconstruction) and LLM-derived semantic relations
    (classification). It serves as a frozen feature extractor during RL.
    """

    # Architecture
    encoder_hidden: List[int] = field(default_factory=lambda: [256, 128])
    decoder_hidden: List[int] = field(default_factory=lambda: [128, 256])
    latent_dim: int = 64
    semantic_classes: int = (
        5  # [No-Relation, Target, Contest, Avoid, Separate] per neighbor
    )

    # Loss weights
    lambda_sem: float = 1.0  # weight of semantic cross-entropy loss
    beta: float = 0.1  # weight of KL divergence regularizer

    # Pretraining
    pretrain_epochs: int = 200
    pretrain_lr: float = 1e-3
    pretrain_batch_size: int = 128
    replay_buffer_size: int = 100_000  # capacity for (obs, sem_label) pairs
    pretrain_rollout_steps: int = 50_000  # env steps to collect before training

    # Semantic label source
    use_llm: bool = False  # True = call LLM API; False = heuristic rules
    llm_model: str = "gpt-4o"  # model identifier for the LLM backend
    llm_api_key: str = ""  # API key (or read from env var OPENAI_API_KEY)


@dataclass
class SemGATConfig:
    """Semantic Graph Attention Network configuration.

    The GAT policy replaces the standard MLP actor/critic, using
    semantic relation types to weight attention over agent neighbors.
    """

    # GAT architecture (per the paper: 2 layers, 4 heads, hidden=64)
    gat_hidden_dim: int = 64
    gat_n_heads: int = 4
    gat_n_layers: int = 2
    gat_dropout: float = 0.0

    # Feature source
    use_vae_latent: bool = True  # prepend VAE latent z to raw obs
    vae_checkpoint: str = ""  # path to pretrained DH-VAE weights

    # Graph construction
    edge_types: List[str] = field(
        default_factory=lambda: ["Target", "Contest", "Avoid", "Separate"]
    )
    max_neighbors: int = 10

    # Semantic reward shaping (paper Eq. 4)
    use_semantic_reward: bool = True
    rew_target_approach: float = 2.0  # w_T
    rew_safe_distance: float = 0.5  # w_S


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
    vae: VAEConfig = field(default_factory=VAEConfig)
    semgat: SemGATConfig = field(default_factory=SemGATConfig)

    def __post_init__(self):
        self.obs.max_num_drones = self.env.num_drones
        self.obs.max_num_targets = self.env.num_targets
        self.obs.max_num_obstacles = self.env.num_obstacles
        self.term.max_steps = self.env.max_steps


def get_default_config() -> FullConfig:
    """Return default configuration."""
    return FullConfig()
