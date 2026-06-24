"""
Semantic relation classification for multi-UAV collaborative navigation.

Implements the LLM-Semantic Enhanced framework (Section IV-A of the paper):
    1. Heuristic rule-based classifier (fast, no API dependency)
    2. LLM-based classifier placeholder (GPT/Claude via prompt engineering)

Relation Taxonomy
------------------
For any pair of entities (u, v) where u is the focal agent:

    No-Relation (0)  — no significant interaction
    Target     (1)  — v is a landmark designated for u (attractive)
    Contest    (2)  — multiple agents vie for the same resource v
    Avoid      (3)  — v is a static/dynamic obstacle (repulsive)
    Separate   (4)  — v is another drone within safety zone (repulsive)

The output is a semantic interaction graph G_t = (V, E_t, W_t) where edges
only exist for non-trivial relations, greatly reducing graph complexity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# ── Relation type constants ─────────────────────────────────────────────
NO_RELATION = 0
TARGET = 1
CONTEST = 2
AVOID = 3
SEPARATE = 4

RELATION_NAMES: Dict[int, str] = {
    NO_RELATION: "No-Relation",
    TARGET: "Target",
    CONTEST: "Contest",
    AVOID: "Avoid",
    SEPARATE: "Separate",
}

RELATION_TO_INDEX: Dict[str, int] = {v: k for k, v in RELATION_NAMES.items()}

NUM_RELATION_CLASSES = len(RELATION_NAMES)  # 5


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class SemanticEdge:
    """A directed edge in the semantic interaction graph."""

    src: int  # source agent index
    dst: int  # destination entity index
    relation: int  # one of {0..4}
    confidence: float = 1.0  # confidence score [0, 1]


@dataclass
class SemanticGraph:
    """Sparse semantic interaction graph for one time step."""

    n_agents: int
    n_targets: int
    n_obstacles: int
    edges: List[SemanticEdge]  # list of directed edges

    def adjacency_matrix(self) -> np.ndarray:
        """Return (n_agents, n_agents) adjacency with relation type as value.

        Non-existent edges are encoded as NO_RELATION (0).
        """
        adj = np.zeros((self.n_agents, self.n_agents), dtype=np.int32)
        for e in self.edges:
            if e.dst < self.n_agents:  # agent-to-agent edges
                adj[e.src, e.dst] = e.relation
        return adj

    def edge_index_and_types(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (2, num_edges) edge_index and (num_edges,) relation types.

        Useful as input to PyTorch Geometric / custom GAT.
        """
        if not self.edges:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        edge_index = np.array([[e.src, e.dst] for e in self.edges], dtype=np.int64).T
        edge_types = np.array([e.relation for e in self.edges], dtype=np.int64)
        return edge_index, edge_types

    def num_edges(self) -> int:
        return len(self.edges)


# ── Heuristic rule-based classifier ─────────────────────────────────────


class HeuristicSemanticClassifier:
    """Rule-based semantic relation classifier.

    This provides a fast, deterministic fallback when no LLM API is available.
    The rules mirror the logic described in the paper (Section IV-A.1) but
    use geometric heuristics instead of natural-language reasoning.

    Parameters
    ----------
    safety_radius : float
        Inter-agent safety distance (m).  Pairs closer than this get Separate.
    obstacle_margin : float
        Extra margin added to obstacle radii for Avoid classification.
    contest_distance : float
        Two agents within this distance of the same target trigger Contest.
    """

    def __init__(
        self,
        safety_radius: float = 5.0,
        obstacle_margin: float = 3.0,
        contest_distance: float = 15.0,
    ):
        self.safety_radius = safety_radius
        self.obstacle_margin = obstacle_margin
        self.contest_distance = contest_distance

    def classify(
        self,
        states: np.ndarray,  # (n_agents, 12) — position at [:, :3], velocity at [:, 3:6]
        target_positions: np.ndarray,  # (n_targets, 3)
        target_assignment: np.ndarray,  # (n_agents,) int — assigned target index per agent
        obstacle_positions: np.ndarray,  # (n_obstacles, 3)
        obstacle_radii: np.ndarray,  # (n_obstacles,)
        carry_status: np.ndarray = None,  # (n_agents,) bool
    ) -> SemanticGraph:
        """Classify all pairwise relations and build the semantic graph.

        Returns
        -------
        SemanticGraph
            Sparse graph containing only semantically meaningful edges.
        """
        n_agents = len(states)
        n_targets = len(target_positions)
        n_obstacles = len(obstacle_positions)
        edges: List[SemanticEdge] = []

        positions = states[:, :3]
        velocities = states[:, 3:6]

        # ── Agent → Target relations ──
        # Determine which agent "claims" which target based on proximity
        target_claimed_by: Dict[int, int] = {}  # target_idx -> agent_idx
        agent_dist_to_targets = np.zeros((n_agents, n_targets))
        for i in range(n_agents):
            for j in range(n_targets):
                agent_dist_to_targets[i, j] = float(
                    np.linalg.norm(positions[i] - target_positions[j])
                )

        for j in range(n_targets):
            closest_agent = int(np.argmin(agent_dist_to_targets[:, j]))
            target_claimed_by[j] = closest_agent

        for i in range(n_agents):
            for j in range(n_targets):
                dist = agent_dist_to_targets[i, j]
                # Target: assigned target OR clearly the closest claimant
                if target_assignment is not None and target_assignment[i] == j:
                    edges.append(
                        SemanticEdge(
                            src=i, dst=n_agents + j, relation=TARGET, confidence=1.0
                        )
                    )
                elif target_claimed_by.get(j) == i:
                    # Nearest unclaimed target → Target
                    edges.append(
                        SemanticEdge(
                            src=i,
                            dst=n_agents + j,
                            relation=TARGET,
                            confidence=0.9,
                        )
                    )
                else:
                    # Check for Contest: another agent is closer to this target
                    closest_other = target_claimed_by.get(j, -1)
                    if closest_other >= 0 and closest_other != i:
                        dist_other = agent_dist_to_targets[closest_other, j]
                        if dist < self.contest_distance and dist_other < self.contest_distance:
                            edges.append(
                                SemanticEdge(
                                    src=i,
                                    dst=n_agents + j,
                                    relation=CONTEST,
                                    confidence=0.7,
                                )
                            )

        # ── Agent → Obstacle relations ──
        for i in range(n_agents):
            pos_i = positions[i]
            vel_i = velocities[i]
            speed = float(np.linalg.norm(vel_i))
            vel_dir = vel_i / (speed + 1e-8)  # unit velocity direction

            for k in range(n_obstacles):
                obs_pos = obstacle_positions[k]
                rel_vec = obs_pos - pos_i
                dist = float(np.linalg.norm(rel_vec))
                effective_radius = obstacle_radii[k] + self.obstacle_margin

                # Avoid if within effective radius
                if dist < effective_radius + self.safety_radius:
                    # Check if obstacle lies within ±45° of velocity direction
                    if speed > 0.5:
                        cos_angle = float(np.dot(vel_dir, rel_vec / (dist + 1e-8)))
                        if cos_angle > 0.7:  # ~45° cone
                            edges.append(
                                SemanticEdge(
                                    src=i,
                                    dst=n_agents + n_targets + k,
                                    relation=AVOID,
                                    confidence=min(1.0, effective_radius / (dist + 1e-8)),
                                )
                            )
                            continue
                    # If not in velocity cone, still Avoid if very close
                    if dist < effective_radius:
                        edges.append(
                            SemanticEdge(
                                src=i,
                                dst=n_agents + n_targets + k,
                                relation=AVOID,
                                confidence=1.0,
                            )
                        )

        # ── Agent → Agent relations (Separate) ──
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                if dist < self.safety_radius:
                    edges.append(
                        SemanticEdge(
                            src=i, dst=j, relation=SEPARATE, confidence=1.0
                        )
                    )
                    edges.append(
                        SemanticEdge(
                            src=j, dst=i, relation=SEPARATE, confidence=1.0
                        )
                    )

        return SemanticGraph(
            n_agents=n_agents,
            n_targets=n_targets,
            n_obstacles=n_obstacles,
            edges=edges,
        )


# ── LLM-based classifier (placeholder) ──────────────────────────────────


# Prompt template adapted from the paper (Section IV-A.2)
_SEMANTIC_PROMPT_TEMPLATE = """You are a Semantic Relation Identifier for a multi-UAV navigation system.
Your task is to analyze the spatial configuration of UAVs, targets, and obstacles,
then output the semantic interaction type for each relevant entity pair.

## Context
- Number of UAVs: {n_agents}
- Number of target landmarks: {n_targets}
- Number of obstacles: {n_obstacles}
- Environment bounds: {bounds}

## Agent States (position [x,y,z], velocity [vx,vy,vz])
{agent_states}

## Target Positions [x,y,z] with assignment status
{target_info}

## Obstacle Positions [x,y,z] with radius
{obstacle_info}

## Task
For each UAV, classify its relationship to every other entity into ONE of:

- **Target (T)**: The entity is a landmark that this UAV should navigate toward.
  - Assigned target, or nearest unvisited landmark within velocity projection.
- **Contest (C)**: Multiple UAVs are competing for the same landmark.
  - Another UAV is also approaching this target with similar priority.
- **Avoid (A)**: The entity is an obstacle that poses a collision risk.
  - Obstacle intersects the UAV's predicted trajectory or lies within safety zone.
- **Separate (S)**: The entity is another UAV within the safety separation zone.
  - Inter-agent distance below safe threshold.
- **No-Relation (0)**: None of the above applies.

## Output Format
Return a JSON object with the following structure:
```json
{{
  "edges": [
    {{"src": <agent_idx>, "dst": <entity_idx>, "relation": "<T|C|A|S>", "confidence": <0.0-1.0>}},
    ...
  ]
}}
```

Entity indexing:
- Agents: 0 to {n_agents_minus_1}
- Targets: {n_agents} to {targets_max}
- Obstacles: {obstacles_min} to {obstacles_max}
"""


class LLMSemanticClassifier:
    """LLM-based semantic relation classifier (placeholder).

    Uses a pre-trained LLM (GPT-4, Claude, etc.) to perform zero-shot
    semantic relation classification via structured prompt engineering.

    Currently a skeleton — implement ``_call_llm()`` with your preferred
    LLM backend (OpenAI API, Anthropic API, local model, etc.).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def classify(
        self,
        states: np.ndarray,
        target_positions: np.ndarray,
        target_assignment: np.ndarray,
        obstacle_positions: np.ndarray,
        obstacle_radii: np.ndarray,
        bounds: np.ndarray = None,
        carry_status: np.ndarray = None,
    ) -> SemanticGraph:
        """Classify relations using the LLM.

        Raises
        ------
        NotImplementedError
            If the LLM backend has not been wired up.
        RuntimeError
            If the API call fails.
        """
        n_agents = len(states)
        n_targets = len(target_positions)
        n_obstacles = len(obstacle_positions)

        prompt = self._build_prompt(
            states, target_positions, target_assignment,
            obstacle_positions, obstacle_radii, bounds,
        )

        response_text = self._call_llm(prompt)

        return self._parse_response(
            response_text, n_agents, n_targets, n_obstacles
        )

    def _build_prompt(
        self,
        states: np.ndarray,
        target_positions: np.ndarray,
        target_assignment: np.ndarray,
        obstacle_positions: np.ndarray,
        obstacle_radii: np.ndarray,
        bounds: np.ndarray = None,
    ) -> str:
        """Build the structured prompt from environment state."""
        n_agents = len(states)
        n_targets = len(target_positions)
        n_obstacles = len(obstacle_positions)

        # Format agent states
        agent_lines = []
        for i in range(n_agents):
            pos = states[i, :3]
            vel = states[i, 3:6]
            assigned = (
                int(target_assignment[i]) if target_assignment is not None else -1
            )
            agent_lines.append(
                f"  Agent {i}: pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}], "
                f"vel=[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}], "
                f"assigned_target={assigned}"
            )

        # Format target info
        target_lines = []
        for j in range(n_targets):
            tp = target_positions[j]
            target_lines.append(
                f"  Target {j + n_agents}: pos=[{tp[0]:.1f}, {tp[1]:.1f}, {tp[2]:.1f}]"
            )

        # Format obstacle info
        obstacle_lines = []
        for k in range(n_obstacles):
            op = obstacle_positions[k]
            r = obstacle_radii[k]
            obstacle_lines.append(
                f"  Obstacle {k + n_agents + n_targets}: "
                f"pos=[{op[0]:.1f}, {op[1]:.1f}, {op[2]:.1f}], radius={r:.1f}"
            )

        bounds_str = str(bounds.tolist()) if bounds is not None else "unknown"

        return _SEMANTIC_PROMPT_TEMPLATE.format(
            n_agents=n_agents,
            n_targets=n_targets,
            n_obstacles=n_obstacles,
            bounds=bounds_str,
            agent_states="\n".join(agent_lines),
            target_info="\n".join(target_lines),
            obstacle_info="\n".join(obstacle_lines),
            n_agents_minus_1=n_agents - 1,
            targets_max=n_agents + n_targets - 1,
            obstacles_min=n_agents + n_targets,
            obstacles_max=n_agents + n_targets + n_obstacles - 1,
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API.

        TODO: Implement with your preferred backend.
        Examples:
            - openai.OpenAI(api_key=...).chat.completions.create(...)
            - anthropic.Anthropic(api_key=...).messages.create(...)
            - requests.post("http://localhost:8000/v1/chat/completions", ...)
        """
        # --- Placeholder ---
        raise NotImplementedError(
            "LLM backend not implemented. "
            "Set use_llm=False in VAEConfig to use heuristic rules, "
            "or implement _call_llm() with your LLM API of choice."
        )

    def _parse_response(
        self,
        response_text: str,
        n_agents: int,
        n_targets: int,
        n_obstacles: int,
    ) -> SemanticGraph:
        """Parse the LLM JSON response into a SemanticGraph."""
        # Extract JSON block from markdown code fences if present
        text = response_text.strip()
        if "```" in text:
            # Find first ```...``` block
            start = text.find("```") + 3
            # Skip optional language tag
            newline = text.find("\n", start)
            start = newline + 1 if newline != -1 else start
            end = text.find("```", start)
            text = text[start:end].strip() if end != -1 else text[start:].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to find a JSON object anywhere in the text
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse LLM response as JSON: {text[:200]}...")

        edges = []
        for item in data.get("edges", []):
            rel_str = item.get("relation", "0")
            if isinstance(rel_str, str):
                rel = RELATION_TO_INDEX.get(rel_str, NO_RELATION)
            else:
                rel = int(rel_str)
            edges.append(
                SemanticEdge(
                    src=int(item["src"]),
                    dst=int(item["dst"]),
                    relation=rel,
                    confidence=float(item.get("confidence", 1.0)),
                )
            )

        return SemanticGraph(
            n_agents=n_agents,
            n_targets=n_targets,
            n_obstacles=n_obstacles,
            edges=edges,
        )


# ── Convenience factory ─────────────────────────────────────────────────


def make_semantic_classifier(
    use_llm: bool = False,
    llm_model: str = "gpt-4o",
    llm_api_key: str = "",
    safety_radius: float = 5.0,
    obstacle_margin: float = 3.0,
    contest_distance: float = 15.0,
) -> Union[HeuristicSemanticClassifier, LLMSemanticClassifier]:
    """Create a semantic classifier instance.

    Parameters
    ----------
    use_llm : bool
        If True, return LLMSemanticClassifier (requires API implementation).
        If False, return HeuristicSemanticClassifier (ready to use).

    Returns
    -------
    HeuristicSemanticClassifier or LLMSemanticClassifier
    """
    if use_llm:
        return LLMSemanticClassifier(model=llm_model, api_key=llm_api_key)
    return HeuristicSemanticClassifier(
        safety_radius=safety_radius,
        obstacle_margin=obstacle_margin,
        contest_distance=contest_distance,
    )


def build_semantic_labels(
    graph: SemanticGraph,
    max_neighbors: int = 10,
) -> np.ndarray:
    """Convert a SemanticGraph into fixed-size label matrix for VAE training.

    Returns
    -------
    labels : np.ndarray, shape (n_agents, max_neighbors, NUM_RELATION_CLASSES)
        One-hot semantic relation labels per agent-neighbor pair.
        Padded with No-Relation class for empty neighbor slots.
    """
    n_agents = graph.n_agents
    labels = np.zeros((n_agents, max_neighbors, NUM_RELATION_CLASSES), dtype=np.float32)
    labels[:, :, NO_RELATION] = 1.0  # default: No-Relation

    # Count neighbors per agent
    neighbor_count = {i: 0 for i in range(n_agents)}

    for edge in graph.edges:
        src = edge.src
        slot = neighbor_count.get(src, 0)
        if slot >= max_neighbors:
            continue
        labels[src, slot, :] = 0.0
        labels[src, slot, edge.relation] = 1.0
        neighbor_count[src] = slot + 1

    return labels
