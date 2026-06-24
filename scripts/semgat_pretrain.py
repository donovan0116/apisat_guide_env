"""
Phase 1: Dual-Head VAE Pretraining for SemGAT-MARL.

Algorithm 1 (lines 1-5) of the paper:
    1. Rollout with random policy to collect (observation, semantic label) pairs
    2. Train the DH-VAE to jointly reconstruct physical state and predict
       semantic relations
    3. Save the frozen encoder for downstream GAT-MAPPO training

Usage:
    # Heuristic semantic labels (default, no API key needed)
    python scripts/semgat_pretrain.py --num_drones 4 --num_targets 4 --total_steps 50000

    # With LLM semantic labels (requires API key)
    python scripts/semgat_pretrain.py --use_llm --llm_model gpt-4o

    # Custom VAE architecture
    python scripts/semgat_pretrain.py --latent_dim 128 --beta 0.05 --lambda_sem 2.0
"""

import argparse
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.semantic import (
    HeuristicSemanticClassifier,
    build_semantic_labels,
    make_semantic_classifier,
)
from core.vae import DualHeadVAE, ReplayBuffer, train_vae_epoch
from envs import ParallelQuadrotorDelivery
from utils.config import FullConfig


def collect_pretraining_data(
    env: ParallelQuadrotorDelivery,
    classifier: HeuristicSemanticClassifier,
    buffer: ReplayBuffer,
    total_steps: int,
    use_llm: bool = False,
    llm_classifier=None,
) -> int:
    """Rollout random policy and collect (obs, semantic_label) pairs.

    Returns the number of transitions collected.
    """
    print(f"[Pretrain] Collecting {total_steps} environment steps...")
    obs_dict, _ = env.reset()
    base_env = env.env  # underlying QuadrotorDeliveryEnv
    n_agents = len(env.possible_agents)
    collected = 0
    start_time = time.time()

    # Helper to extract state from the underlying environment
    def _get_env_state():
        target_positions = np.array([t.position for t in base_env._targets])
        return {
            "states": base_env._states.copy(),
            "target_positions": target_positions,
            "target_assignment": base_env._target_assignment.copy(),
            "obstacle_positions": base_env._obstacle_positions.copy(),
            "obstacle_radii": base_env._obstacle_radii.copy(),
            "carry_status": base_env._carry_status.copy() if base_env._carry_status is not None else None,
        }

    while collected < total_steps:
        # Random actions
        actions = {
            agent_id: np.random.uniform(-1, 1, size=4)
            for agent_id in env.agents
        }

        next_obs, rewards, terms, truncs, next_info = env.step(actions)

        # Build semantic graph from current underlying env state
        state_data = _get_env_state()

        if use_llm and llm_classifier is not None:
            graph = llm_classifier.classify(
                state_data["states"],
                state_data["target_positions"],
                state_data["target_assignment"],
                state_data["obstacle_positions"],
                state_data["obstacle_radii"],
                carry_status=state_data["carry_status"],
            )
        else:
            graph = classifier.classify(
                state_data["states"],
                state_data["target_positions"],
                state_data["target_assignment"],
                state_data["obstacle_positions"],
                state_data["obstacle_radii"],
                carry_status=state_data["carry_status"],
            )

        # Convert graph to fixed-size label matrix
        sem_labels = build_semantic_labels(graph, max_neighbors=buffer.max_neighbors)

        # Collect per-agent observations
        agent_ids = env.possible_agents
        obs_batch = np.zeros((n_agents, buffer.obs_dim), dtype=np.float32)
        for i, aid in enumerate(agent_ids):
            obs_batch[i] = obs_dict.get(aid, np.zeros(buffer.obs_dim, dtype=np.float32))

        buffer.add(obs_batch, sem_labels)
        collected += 1

        obs_dict = next_obs

        if not env.agents:
            obs_dict, _ = env.reset()

        if collected % 5000 == 0:
            elapsed = time.time() - start_time
            fps = collected / max(elapsed, 1e-6)
            print(f"  collected {collected}/{total_steps} steps, {fps:.0f} steps/s")

    env.close()
    print(f"[Pretrain] Data collection complete: {buffer.size} transitions")
    return buffer.size


def train_vae(
    vae: DualHeadVAE,
    buffer: ReplayBuffer,
    epochs: int,
    batch_size: int,
    device: torch.device,
    log_dir: str = "",
) -> Dict[str, float]:
    """Train the DH-VAE on collected data.

    Returns final epoch loss dict.
    """
    writer = SummaryWriter(log_dir) if SummaryWriter and log_dir else None
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    final_losses = {}

    print(f"[Pretrain] Training VAE for {epochs} epochs (batch={batch_size})...")
    for epoch in range(epochs):
        losses = train_vae_epoch(vae, buffer, optimizer, batch_size, device)

        if writer:
            for k, v in losses.items():
                writer.add_scalar(f"vae/{k}", v, epoch)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"  epoch {epoch:>4d}/{epochs} | "
                f"total={losses['total']:.4f} "
                f"phys={losses['phys']:.4f} "
                f"sem={losses['sem']:.4f} "
                f"kl={losses['kl']:.4f}"
            )
        final_losses = losses

    if writer:
        writer.close()
    return final_losses


def build_config(args) -> FullConfig:
    """Build configuration from CLI arguments."""
    cfg = FullConfig()
    cfg.env.num_drones = args.num_drones
    cfg.env.num_targets = args.num_targets
    cfg.env.num_obstacles = args.num_obstacles
    cfg.env.task_mode = args.task_mode
    cfg.env.aggregate_phy_steps = args.aggregate_phy_steps
    cfg.env.max_steps = args.max_episode_steps
    cfg.obs.max_num_drones = args.num_drones
    cfg.obs.max_num_targets = args.num_targets
    cfg.obs.max_num_obstacles = args.num_obstacles
    cfg.term.max_steps = args.max_episode_steps

    # VAE config
    cfg.vae.latent_dim = args.latent_dim
    cfg.vae.beta = args.beta
    cfg.vae.lambda_sem = args.lambda_sem
    cfg.vae.pretrain_epochs = args.vae_epochs
    cfg.vae.pretrain_lr = args.vae_lr
    cfg.vae.pretrain_batch_size = args.batch_size
    cfg.vae.pretrain_rollout_steps = args.total_steps
    cfg.vae.replay_buffer_size = args.buffer_size
    cfg.vae.use_llm = args.use_llm
    cfg.vae.llm_model = args.llm_model
    cfg.vae.llm_api_key = args.llm_api_key

    # SemGAT config
    cfg.semgat.gat_hidden_dim = args.gat_hidden
    cfg.semgat.gat_n_heads = args.gat_heads
    cfg.semgat.gat_n_layers = args.gat_layers
    cfg.semgat.vae_checkpoint = args.output

    cfg.train.seed = args.seed
    cfg.train.device = args.device
    cfg.train.log_dir = args.log_dir
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: Pretrain DH-VAE for SemGAT-MARL"
    )

    # Environment
    parser.add_argument("--num_drones", type=int, default=4)
    parser.add_argument("--num_targets", type=int, default=4)
    parser.add_argument("--num_obstacles", type=int, default=10)
    parser.add_argument("--task_mode", choices=["reach", "delivery"], default="reach")
    parser.add_argument("--aggregate_phy_steps", type=int, default=20)
    parser.add_argument("--max_episode_steps", type=int, default=1000)

    # Data collection
    parser.add_argument("--total_steps", type=int, default=50_000,
                        help="Environment steps to collect for pretraining")
    parser.add_argument("--buffer_size", type=int, default=100_000)

    # VAE architecture
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL divergence weight")
    parser.add_argument("--lambda_sem", type=float, default=1.0,
                        help="Semantic loss weight")

    # VAE training
    parser.add_argument("--vae_epochs", type=int, default=200)
    parser.add_argument("--vae_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)

    # GAT (stored in checkpoint for downstream use)
    parser.add_argument("--gat_hidden", type=int, default=64)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--gat_layers", type=int, default=2)

    # LLM
    parser.add_argument("--use_llm", action="store_true",
                        help="Use LLM for semantic labeling (requires API key)")
    parser.add_argument("--llm_model", type=str, default="gpt-4o")
    parser.add_argument("--llm_api_key", type=str, default="")

    # Output
    parser.add_argument("--output", type=str, default="models/semgat/vae_pretrained.pt")
    parser.add_argument("--log_dir", type=str, default="logs/semgat_pretrain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device(args.device)

    # Create probe environment to get observation dimensions
    probe_env = ParallelQuadrotorDelivery(
        num_drones=cfg.env.num_drones,
        num_targets=cfg.env.num_targets,
        num_obstacles=cfg.env.num_obstacles,
        task_mode=cfg.env.task_mode,
        bounds=cfg.env.bounds_array,
        quad_params=cfg.quad,
        obs_config=cfg.obs,
        act_config=cfg.act,
        reward_config=cfg.reward,
        term_config=cfg.term,
        aggregate_phy_steps=cfg.env.aggregate_phy_steps,
        seed=cfg.train.seed,
    )
    agent0 = probe_env.possible_agents[0]
    obs_dim = probe_env.observation_space(agent0).shape[0]
    probe_env.close()
    print(f"[Pretrain] Observation dim: {obs_dim}")

    # Create classifier
    classifier = make_semantic_classifier(
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
    )
    llm_classifier = classifier if args.use_llm else None
    if not args.use_llm:
        print("[Pretrain] Using heuristic semantic classifier")

    # Create buffer and collect data
    max_neighbors = cfg.semgat.max_neighbors
    buffer = ReplayBuffer(
        obs_dim=obs_dim,
        max_neighbors=max_neighbors,
        num_classes=5,
        capacity=cfg.vae.replay_buffer_size,
    )

    env = ParallelQuadrotorDelivery(
        num_drones=cfg.env.num_drones,
        num_targets=cfg.env.num_targets,
        num_obstacles=cfg.env.num_obstacles,
        task_mode=cfg.env.task_mode,
        bounds=cfg.env.bounds_array,
        quad_params=cfg.quad,
        obs_config=cfg.obs,
        act_config=cfg.act,
        reward_config=cfg.reward,
        term_config=cfg.term,
        aggregate_phy_steps=cfg.env.aggregate_phy_steps,
        seed=cfg.train.seed,
    )

    collect_pretraining_data(
        env, classifier, buffer,
        total_steps=cfg.vae.pretrain_rollout_steps,
        use_llm=args.use_llm,
        llm_classifier=llm_classifier,
    )

    if buffer.size < cfg.vae.pretrain_batch_size:
        print(
            f"[Pretrain] ERROR: buffer too small ({buffer.size} < {cfg.vae.pretrain_batch_size}). "
            "Increase --total_steps."
        )
        return

    # Create and train VAE
    vae = DualHeadVAE(
        obs_dim=obs_dim,
        latent_dim=cfg.vae.latent_dim,
        max_neighbors=max_neighbors,
        num_classes=5,
        encoder_hidden=cfg.vae.encoder_hidden,
        decoder_hidden=cfg.vae.decoder_hidden,
        beta=cfg.vae.beta,
        lambda_sem=cfg.vae.lambda_sem,
    ).to(device)

    print(f"[Pretrain] VAE architecture:\n{vae}")
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"[Pretrain] Total parameters: {n_params:,}")

    final_losses = train_vae(
        vae, buffer,
        epochs=cfg.vae.pretrain_epochs,
        batch_size=cfg.vae.pretrain_batch_size,
        device=device,
        log_dir=args.log_dir,
    )

    # Save checkpoint
    checkpoint = {
        "vae_state_dict": vae.state_dict(),
        "obs_dim": obs_dim,
        "latent_dim": cfg.vae.latent_dim,
        "max_neighbors": max_neighbors,
        "config": {
            "vae": asdict(cfg.vae),
            "semgat": asdict(cfg.semgat),
            "env": asdict(cfg.env),
        },
        "final_losses": {k: float(v) for k, v in final_losses.items()},
    }
    torch.save(checkpoint, args.output)
    print(f"[Pretrain] VAE checkpoint saved to {args.output}")
    print(f"[Pretrain] Final losses: {final_losses}")


if __name__ == "__main__":
    main()
