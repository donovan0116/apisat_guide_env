"""
Dual-Head Variational Autoencoder (DH-VAE) for semantic-physical fusion.

Section IV-B of the paper: the DH-VAE learns a joint latent representation z
that encodes both physical dynamics (reconstruction) and task-level semantic
relations (classification).  The learned z serves as a frozen feature extractor
during downstream GAT-MAPPO training.

Architecture
------------
    Encoder:   obs x → [256, 128] → μ, σ → z (latent_dim)
    ├── Physical Decoder:  z → [128, 256] → x̂        (MSE)
    └── Semantic Decoder:  z → [128, 64] → ŷ_sem      (CrossEntropy)

Loss (Eq. 3 in the paper)
-------------------------
    L_total = ||x - x̂||² + λ_sem · CE(y_sem, ŷ_sem) + β · KL[ q(z|x) || N(0,I) ]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


# ── Building blocks ──────────────────────────────────────────────────────


def _build_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    activation: str = "leaky_relu",
    final_activation: bool = False,
) -> nn.Sequential:
    """Construct an MLP with configurable hidden layers."""
    layers: List[nn.Module] = []
    prev = in_dim
    act_cls = nn.LeakyReLU if activation == "leaky_relu" else nn.ReLU
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if final_activation:
        layers.append(act_cls())
    return nn.Sequential(*layers)


# ── DH-VAE ───────────────────────────────────────────────────────────────


class DHVAEEncoder(nn.Module):
    """Encoder: observation → latent distribution parameters (μ, log σ²)."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.shared = _build_mlp(obs_dim, hidden_dims[:-1], hidden_dims[-1])
        self.mu_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class DHVAEPhysicalDecoder(nn.Module):
    """Physical reconstruction head: latent z → reconstructed observation x̂."""

    def __init__(
        self,
        latent_dim: int = 64,
        obs_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 256]
        self.net = _build_mlp(latent_dim, hidden_dims, obs_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DHVAESemanticDecoder(nn.Module):
    """Semantic prediction head: latent z → relation class logits per neighbor.

    Output shape: (batch, max_neighbors, num_classes).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        max_neighbors: int = 10,
        num_classes: int = 5,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        out_dim = max_neighbors * num_classes
        self.net = _build_mlp(latent_dim, hidden_dims, out_dim)
        self.max_neighbors = max_neighbors
        self.num_classes = num_classes

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return logits shaped (batch, max_neighbors, num_classes)."""
        flat = self.net(z)
        return flat.view(z.shape[0], self.max_neighbors, self.num_classes)


class DualHeadVAE(nn.Module):
    """Dual-Head Variational Autoencoder.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the flattened local observation.
    latent_dim : int
        Size of the latent code z.
    max_neighbors : int
        Max number of neighbor slots for semantic prediction.
    num_classes : int
        Number of semantic relation classes (default 5).
    encoder_hidden : list[int]
        Hidden layer sizes for the encoder MLP.
    decoder_hidden : list[int]
        Hidden layer sizes for the decoder MLPs.
    beta : float
        Weight of the KL regularizer (paper β).
    lambda_sem : float
        Weight of the semantic cross-entropy loss (paper λ_sem).
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 64,
        max_neighbors: int = 10,
        num_classes: int = 5,
        encoder_hidden: Optional[List[int]] = None,
        decoder_hidden: Optional[List[int]] = None,
        beta: float = 0.1,
        lambda_sem: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.max_neighbors = max_neighbors
        self.num_classes = num_classes
        self.beta = beta
        self.lambda_sem = lambda_sem

        # Sub-modules
        self.encoder = DHVAEEncoder(obs_dim, latent_dim, encoder_hidden)
        self.phys_decoder = DHVAEPhysicalDecoder(latent_dim, obs_dim, decoder_hidden)
        self.sem_decoder = DHVAESemanticDecoder(
            latent_dim, max_neighbors, num_classes,
            hidden_dims=[decoder_hidden[0], decoder_hidden[0] // 2] if decoder_hidden else None,
        )

    def encode(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode observation to latent distribution.

        Returns
        -------
        z : (batch, latent_dim) sampled latent code
        mu : (batch, latent_dim) posterior mean
        logvar : (batch, latent_dim) posterior log-variance
        """
        mu, logvar = self.encoder(x)
        if deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        return z, mu, logvar

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns
        -------
        x_recon : (batch, obs_dim)
        sem_logits : (batch, max_neighbors, num_classes)
        mu : (batch, latent_dim)
        logvar : (batch, latent_dim)
        """
        z, mu, logvar = self.encode(x, deterministic=False)
        x_recon = self.phys_decoder(z)
        sem_logits = self.sem_decoder(z)
        return x_recon, sem_logits, mu, logvar

    def compute_loss(
        self,
        x: torch.Tensor,
        sem_labels: torch.Tensor,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """Compute the composite VAE loss (paper Eq. 3).

        Parameters
        ----------
        x : (batch, obs_dim) raw observations
        sem_labels : (batch, max_neighbors, num_classes) one-hot semantic labels
        reduction : "mean" | "sum" | "none"

        Returns
        -------
        dict with keys: total, phys, sem, kl
        """
        x_recon, sem_logits, mu, logvar = self.forward(x)

        # Physical reconstruction loss (MSE)
        phys_loss = F.mse_loss(x_recon, x, reduction=reduction)

        # Semantic classification loss (cross-entropy over neighbor slots)
        # sem_logits: (B, K, C), sem_labels: (B, K, C)
        sem_loss = F.cross_entropy(
            sem_logits.reshape(-1, self.num_classes),
            sem_labels.reshape(-1, self.num_classes).argmax(dim=-1),
            reduction=reduction,
        )

        # KL divergence to standard normal prior
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        if reduction == "mean":
            kl = kl.mean()
        elif reduction == "sum":
            kl = kl.sum()

        total = phys_loss + self.lambda_sem * sem_loss + self.beta * kl

        return {
            "total": total,
            "phys": phys_loss,
            "sem": sem_loss,
            "kl": kl,
        }

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent features z (used during RL training).

        The encoder is called deterministically (z = μ).
        """
        z, _, _ = self.encode(x, deterministic=True)
        return z


# ── Training utilities ───────────────────────────────────────────────────


class ReplayBuffer:
    """Simple replay buffer for VAE pre-training data."""

    def __init__(self, obs_dim: int, max_neighbors: int, num_classes: int, capacity: int = 100_000):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.max_neighbors = max_neighbors
        self.num_classes = num_classes

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.sem_labels = np.zeros(
            (capacity, max_neighbors, num_classes), dtype=np.float32
        )
        self._ptr = 0
        self._full = False

    @property
    def size(self) -> int:
        return self.capacity if self._full else self._ptr

    def add(self, obs_batch: np.ndarray, sem_batch: np.ndarray):
        """Add a batch of transitions to the buffer."""
        n = obs_batch.shape[0]
        assert n <= self.capacity, f"Batch size {n} exceeds capacity {self.capacity}"

        end = min(self._ptr + n, self.capacity)
        space = end - self._ptr
        self.obs[self._ptr : end] = obs_batch[:space]
        self.sem_labels[self._ptr : end] = sem_batch[:space]

        if space < n:
            # Wrap around
            remaining = n - space
            self.obs[:remaining] = obs_batch[space:]
            self.sem_labels[:remaining] = sem_batch[space:]
            self._full = True
        self._ptr = (self._ptr + n) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample a batch."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[indices], dtype=torch.float32),
            torch.as_tensor(self.sem_labels[indices], dtype=torch.float32),
        )


def train_vae_epoch(
    vae: DualHeadVAE,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch over the replay buffer.

    Returns averaged loss components for logging.
    """
    vae.train()
    n_batches = max(1, buffer.size // batch_size)
    losses: Dict[str, List[float]] = {"total": [], "phys": [], "sem": [], "kl": []}

    for _ in range(n_batches):
        obs_batch, sem_batch = buffer.sample(batch_size)
        obs_batch = obs_batch.to(device)
        sem_batch = sem_batch.to(device)

        loss_dict = vae.compute_loss(obs_batch, sem_batch)
        optimizer.zero_grad(set_to_none=True)
        loss_dict["total"].backward()
        optimizer.step()

        for k in losses:
            losses[k].append(float(loss_dict[k].item()))

    return {k: float(np.mean(v)) for k, v in losses.items()}
