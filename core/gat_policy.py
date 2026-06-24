"""
Graph Attention Network (GAT) policy for SemGAT-MARL.

Section IV-C of the paper: the GAT-based policy replaces the standard MLP
actor/critic, using semantic relation types to weight attention over neighbors.

Architecture (per the paper)
-----------------------------
- 2 GAT layers with 4 attention heads and hidden dimension 64
- Input: latent vector z_t (from frozen DH-VAE) + raw agent states
- Attention coefficients incorporate one-hot semantic relation type r_ij
- Actor: GAT → action mean + log_std (decentralized)
- Critic: GAT → global pooling → scalar value (centralized)

Key equation (paper Eq. 5-6):
    α_ij = softmax( LeakyReLU( a^T [W h_i || W h_j || r_ij] ) )
    h_i' = σ( Σ_j α_ij W h_j )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Reuse the log-prob helper from mappo_train
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
TANH_EPS = 1e-6


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + TANH_EPS, 1.0 - TANH_EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


# ── GAT layer with semantic relation injection ───────────────────────────


class SemanticGATLayer(nn.Module):
    """Single GAT layer with semantic relation-aware attention.

    Injects one-hot relation type r_ij into the attention computation
    so that the model learns to prioritize different interaction types
    (e.g., Target > Avoid > Separate).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        num_edge_types: int = 5,
        dropout: float = 0.0,
        concat: bool = True,  # concat heads (True for hidden, False for output)
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout

        # Linear projection for node features (per head)
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)

        # Attention mechanism: score = a^T [W h_i || W h_j || r_ij]
        # Per-head attention vector: 2*out_dim (for h_i, h_j) + num_edge_types (for r_ij)
        attn_dim = 2 * out_dim + num_edge_types
        self.attn = nn.Parameter(torch.empty(n_heads, attn_dim))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))

        # LeakyReLU slope
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Bias
        if concat:
            self.bias = nn.Parameter(torch.zeros(n_heads * out_dim))
        else:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self._alpha: Optional[torch.Tensor] = None  # stored attention for analysis

    @property
    def attention_weights(self) -> Optional[torch.Tensor]:
        """Last computed attention coefficients (for interpretability)."""
        return self._alpha

    def forward(
        self,
        x: torch.Tensor,  # (N, in_dim) node features
        edge_index: torch.Tensor,  # (2, E) edges in COO format
        edge_type: torch.Tensor,  # (E,) relation type per edge
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (N, in_dim)
            Node features for N agents.
        edge_index : (2, E)
            Source and destination indices for E edges.
        edge_type : (E,)
            Integer relation type for each edge (0..4).

        Returns
        -------
        out : (N, n_heads * out_dim) if concat else (N, out_dim)
        """
        N = x.shape[0]
        E = edge_index.shape[1]

        # Linear projection: (N, n_heads * out_dim) → (N, n_heads, out_dim)
        Wh = self.W(x).view(N, self.n_heads, self.out_dim)

        # Gather source and destination features
        src, dst = edge_index[0], edge_index[1]
        Wh_src = Wh[src]  # (E, n_heads, out_dim)
        Wh_dst = Wh[dst]  # (E, n_heads, out_dim)

        # One-hot encode edge types: (E,) → (E, num_edge_types)
        edge_type_oh = F.one_hot(edge_type.long(), num_classes=self.attn.shape[1] - 2 * self.out_dim).float()

        # Concatenate [Wh_src || Wh_dst || r_ij]: (E, n_heads, 2*out_dim + num_edge_types)
        attn_input = torch.cat([Wh_src, Wh_dst, edge_type_oh.unsqueeze(1).expand(-1, self.n_heads, -1)], dim=-1)

        # Compute attention scores: (E, n_heads)
        e = (attn_input * self.attn.unsqueeze(0)).sum(dim=-1)
        e = self.leaky_relu(e)

        # Build sparse attention matrix α[dst, head, src] = e[edge, head]
        # Then softmax over destination nodes per source per head.
        alpha_vals = torch.full((N, self.n_heads, N), float("-inf"), device=x.device)
        for h in range(self.n_heads):
            alpha_vals[dst, h, src] = e[:, h]

        # Softmax over the last dim (incoming edges per source)
        alpha_vals = F.softmax(alpha_vals, dim=-1)  # (N, n_heads, N)
        alpha_vals = torch.nan_to_num(alpha_vals, nan=0.0)

        if self.dropout > 0:
            alpha_vals = F.dropout(alpha_vals, p=self.dropout, training=self.training)

        self._alpha = alpha_vals.detach()

        # Aggregate: h_i' = σ( Σ_j α_ij W h_j )
        # α_vals: (N, n_heads, N), Wh: (N, n_heads, out_dim)
        # For each head h: α[:,h,:] @ Wh[:,h,:] → (N, out_dim)
        out_chunks = []
        for h in range(self.n_heads):
            out_h = alpha_vals[:, h, :] @ Wh[:, h, :]  # (N, N) @ (N, out_dim) → (N, out_dim)
            out_chunks.append(out_h)
        out = torch.stack(out_chunks, dim=1)  # (N, n_heads, out_dim)

        if self.concat:
            out = out.reshape(N, self.n_heads * self.out_dim)
        else:
            out = out.mean(dim=1)  # average over heads

        out = out + self.bias
        return F.elu(out)


# ── GAT Actor ────────────────────────────────────────────────────────────


class GATActor(nn.Module):
    """GAT-based decentralized actor with semantic relation-aware attention.

    Replaces the standard MLP actor in MAPPO.  Input includes:
        - Latent features z from frozen DH-VAE (optional)
        - Raw local observation of the agent
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        latent_dim: int = 64,
        use_latent: bool = True,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 2,
        num_edge_types: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_latent = use_latent

        # Input dim per node: latent_z + agent_own_state (pos+vel=6) + target_info snippet
        # We use the full local obs as node features for simplicity
        node_in_dim = latent_dim + obs_dim if use_latent else obs_dim

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for layer_idx in range(gat_layers):
            is_last = layer_idx == gat_layers - 1
            self.gat_layers.append(
                SemanticGATLayer(
                    in_dim=node_in_dim if layer_idx == 0 else gat_hidden * gat_heads,
                    out_dim=gat_hidden,
                    n_heads=gat_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                    concat=not is_last,  # concat heads for hidden, average for output
                )
            )

        # Output head: GAT output → action mean + log_std
        # Last GAT layer always has concat=False, so output dim = gat_hidden
        final_dim = gat_hidden
        self.mean_head = nn.Linear(final_dim, act_dim)
        self.log_std_head = nn.Linear(final_dim, act_dim)

        # Initialize output heads
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, -0.5)

    def forward(
        self,
        obs: torch.Tensor,  # (N, obs_dim) node features
        latent_z: torch.Tensor,  # (N, latent_dim) VAE features (optional)
        edge_index: torch.Tensor,  # (2, E)
        edge_type: torch.Tensor,  # (E,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and log_std.

        Returns
        -------
        mean : (N, act_dim)
        log_std : (N, act_dim)
        """
        if self.use_latent and latent_z is not None:
            x = torch.cat([latent_z, obs], dim=-1)
        else:
            x = obs

        for gat in self.gat_layers:
            x = gat(x, edge_index, edge_type)

        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        latent_z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions with tanh squashing.

        Returns
        -------
        action : (N, act_dim) in [-1, 1]
        log_prob : (N,) log-prob of the action
        entropy : (N,) policy entropy
        """
        mean, log_std = self.forward(obs, latent_z, edge_index, edge_type)
        std = log_std.exp()
        dist = Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = _log_prob_from_pre_tanh(dist, pre_tanh, action)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        latent_z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-prob of given actions (for PPO update)."""
        mean, log_std = self.forward(obs, latent_z, edge_index, edge_type)
        std = log_std.exp()
        dist = Normal(mean, std)
        clipped_actions = torch.clamp(actions, -1.0 + TANH_EPS, 1.0 - TANH_EPS)
        pre_tanh = atanh(clipped_actions)
        log_prob = _log_prob_from_pre_tanh(dist, pre_tanh, clipped_actions)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    @property
    def attention_weights(self) -> List[Optional[torch.Tensor]]:
        """Return attention weights from all layers (for interpretability)."""
        return [gat.attention_weights for gat in self.gat_layers]


# ── GAT Critic ───────────────────────────────────────────────────────────


class GATCritic(nn.Module):
    """GAT-based centralized critic.

    Processes the full interaction graph then pools node features
    into a scalar team value estimate.
    """

    def __init__(
        self,
        global_dim: int,
        latent_dim: int = 64,
        use_latent: bool = True,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 2,
        num_edge_types: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_latent = use_latent

        # The critic receives the full global observation concatenated per-agent
        # For simplicity, we use the global_obs directly with an MLP on top of GAT output
        node_in_dim = latent_dim + global_dim if use_latent else global_dim

        self.gat_layers = nn.ModuleList()
        for layer_idx in range(gat_layers):
            is_last = layer_idx == gat_layers - 1
            self.gat_layers.append(
                SemanticGATLayer(
                    in_dim=node_in_dim if layer_idx == 0 else gat_hidden * gat_heads,
                    out_dim=gat_hidden,
                    n_heads=gat_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                    concat=not is_last,
                )
            )

        # Global pooling → scalar value
        # Last GAT layer always has concat=False, so output dim = gat_hidden
        final_dim = gat_hidden

        # Flat-mode projection: maps [latent_z || global_obs] → final_dim
        flat_in = (latent_dim if use_latent else 0) + global_dim
        self.flat_proj = nn.Sequential(
            nn.Linear(flat_in, final_dim),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, 1),
        )

        # Initialize weights
        nn.init.orthogonal_(self.flat_proj[0].weight, gain=np.sqrt(2))
        nn.init.constant_(self.flat_proj[0].bias, 0.0)
        nn.init.orthogonal_(self.value_head[0].weight, gain=np.sqrt(2))
        nn.init.constant_(self.value_head[0].bias, 0.0)
        nn.init.orthogonal_(self.value_head[2].weight, gain=0.01)
        nn.init.constant_(self.value_head[2].bias, 0.0)

    def forward(
        self,
        global_obs: torch.Tensor,  # (N, global_dim) if graph mode, or (B, global_dim) if flat
        latent_z: torch.Tensor,  # (N, latent_dim) or (B, latent_dim)
        edge_index: torch.Tensor = None,  # (2, E) or None
        edge_type: torch.Tensor = None,  # (E,) or None
    ) -> torch.Tensor:
        """Estimate team value.

        Two modes:
        1. Graph mode (edge_index is not None): processes per-agent features
           with GAT, then pools to scalar.
        2. Flat mode (edge_index is None): squeezes through value head
           directly — used during PPO mini-batch updates.
        """
        # ── Flat mode (no graph) ──
        if edge_index is None or edge_index.shape[1] == 0:
            if self.use_latent:
                x = torch.cat([latent_z, global_obs], dim=-1)
            else:
                x = global_obs
            # Project to final_dim and estimate per-sample values
            proj = self.flat_proj(x)  # (B, final_dim)
            return self.value_head(proj).squeeze(-1)  # (B,)

        # ── Graph mode ──
        N = latent_z.shape[0]

        # Broadcast global obs to each agent if needed
        if global_obs.ndim == 1:
            global_obs = global_obs.unsqueeze(0).expand(N, -1)
        elif global_obs.shape[0] == 1:
            global_obs = global_obs.expand(N, -1)

        if self.use_latent:
            x = torch.cat([latent_z, global_obs], dim=-1)
        else:
            x = global_obs

        for gat in self.gat_layers:
            x = gat(x, edge_index, edge_type)

        # Pool across agents and project to scalar
        pooled = x.mean(dim=0, keepdim=True)  # (1, final_dim)
        value = self.value_head(pooled).squeeze(-1)  # scalar
        return value


# ── Helpers ──────────────────────────────────────────────────────────────


def _log_prob_from_pre_tanh(
    dist: Normal, pre_tanh: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    """Compute log-prob under tanh-squashed Gaussian (same as mappo_train.py)."""
    correction = torch.log(1.0 - action.pow(2) + TANH_EPS)
    return (dist.log_prob(pre_tanh) - correction).sum(dim=-1)


def build_graph_from_semantic(
    n_agents: int,
    edges: List,  # List of SemanticEdge
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of SemanticEdge to (edge_index, edge_type) tensors.

    Adds self-loops with a special NO_RELATION type so each node
    attends to its own features.
    """
    edge_index_list: List[Tuple[int, int]] = []
    edge_type_list: List[int] = []

    # Add self-loops
    for i in range(n_agents):
        edge_index_list.append((i, i))
        edge_type_list.append(0)  # No-Relation for self

    # Add semantic edges
    for e in edges:
        if e.src < n_agents and e.dst < n_agents:
            # Only include agent→agent edges (Separate) for GAT
            # Agent→Target and Agent→Obstacle edges are encoded
            # in the observation, not in the graph topology
            pass
        edge_index_list.append((e.src, e.dst))
        edge_type_list.append(e.relation)

    if not edge_index_list:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_type = torch.zeros((0,), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long, device=device).T
        edge_type = torch.tensor(edge_type_list, dtype=torch.long, device=device)

    return edge_index, edge_type
