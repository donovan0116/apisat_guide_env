"""
Gymnasium environment: QuadrotorDelivery-v0

Multi-agent quadrotor delivery task assignment with obstacle avoidance.

Observation Space (per-agent, flat):
    - self state: 12 (pos, vel, euler, omega)
    - relative drone info: max_drones * 3
    - target info: max_targets * 4
    - obstacle info: max_obstacles * 3
    - carry status: 1

Action Space (per-agent):
    - Box(4, low=-1, high=1)  [thrust, taux, tauy, tauz] normalized
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym


@dataclass
class _FallbackRenderEvent:
    kind: str
    position: np.ndarray
    drone_id: int = -1
    target_id: int = -1
    timestamp: float = 0.0
    duration: float = 0.6
    color: str = "white"


def _make_event(
    kind: str,
    position: np.ndarray,
    *,
    color: str = "white",
    drone_id: int = -1,
    target_id: int = -1,
    duration: float = 0.6,
):
    """Construct a :class:`utils.rendering.RenderEvent` (lazy import)."""
    try:
        from utils.rendering import RenderEvent
    except ImportError:
        RenderEvent = _FallbackRenderEvent
    return RenderEvent(
        kind=kind,
        position=np.asarray(position, dtype=float),
        drone_id=drone_id,
        target_id=target_id,
        color=color,
        duration=duration,
    )


from gymnasium import spaces

from core.dynamics import QuadrotorDynamics, QuadrotorParams
from core.state import ObsConfig, build_local_obs, build_global_obs
from core.action import ActionConfig, normalize_action
from core.target import (
    Target,
    TargetType,
    Obstacle,
    generate_random_targets,
    generate_random_obstacles,
)
from core.reward import RewardConfig, RewardCalculator
from core.termination import TerminationConfig, TerminationChecker


class QuadrotorDeliveryEnv(gym.Env):
    """
    Multi-agent quadrotor delivery environment.

    Supports two task modes:
        - "reach":     Drones fly directly to target points.
        - "delivery":  Drones pair pickup->delivery targets.

    Supports centralized (flat) observations for CTDE training.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "rgb_array_list", "video", "top_down"],
        "render_fps": 30,
    }

    def __init__(
        self,
        num_drones: int = 4,
        num_targets: int = 4,
        num_obstacles: int = 10,
        task_mode: str = "reach",
        bounds: np.ndarray = None,  # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
        quad_params: QuadrotorParams = None,
        obs_config: ObsConfig = None,
        act_config: ActionConfig = None,
        reward_config: RewardConfig = None,
        term_config: TerminationConfig = None,
        seed: int = None,
        render_mode: str = None,
        aggregate_phy_steps: int = 1,
        device: str = "cpu",
    ):
        super().__init__()

        self.num_drones = num_drones
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        self.task_mode = task_mode
        self.bounds = (
            bounds
            if bounds is not None
            else np.array([[-50.0, 50.0], [-50.0, 50.0], [0.0, 30.0]])
        )
        self.aggregate_phy_steps = aggregate_phy_steps
        self.render_mode = self._normalize_render_mode(render_mode)
        self.device = device

        # --- Sub-modules ---
        self.dynamics = QuadrotorDynamics(quad_params)
        self.obs_config = obs_config or ObsConfig(
            max_num_drones=num_drones,
            max_num_targets=num_targets,
            max_num_obstacles=num_obstacles,
        )
        self.act_config = act_config or ActionConfig()
        self.reward_calc = RewardCalculator(reward_config)
        self.term_checker = TerminationChecker(term_config)

        # --- Spaces ---
        self.agent_obs_dim = self.obs_config.total_dim
        self.agent_act_dim = self.dynamics.action_dim

        self.observation_space = spaces.Dict(
            {
                "agent_obs": spaces.Box(
                    -1.0, 1.0, shape=(num_drones, self.agent_obs_dim), dtype=np.float32
                ),
                "global_obs": spaces.Box(
                    -np.inf, np.inf, shape=(self._global_obs_dim(),), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(num_drones, self.agent_act_dim), dtype=np.float32
        )

        # --- Internal state ---
        self._states: np.ndarray = None  # [n_drones, 12]
        self._targets: List[Target] = None
        self._obstacles: List[Obstacle] = None
        self._obstacle_positions: np.ndarray = None
        self._obstacle_radii: np.ndarray = None
        self._carry_status: np.ndarray = None  # [n_drones] bool
        self._target_assigned: np.ndarray = None  # [n_targets] bool
        self._target_assignment: np.ndarray = None  # [n_drones] int
        self._steps: int = 0
        self._last_rewards: np.ndarray = None
        self._prev_target_assigned: np.ndarray = None
        self._prev_carry_status: np.ndarray = None
        self._collisions_this_step: list = []

        # --- Renderer (lazily created) ---
        self._renderer = None
        self._rgb_buffer: List[np.ndarray] = []
        self._render_event_queue: list = []

        self.np_random = np.random.default_rng(seed)

    # ==================== Gym API ====================

    def reset(
        self, seed: int = None, options: dict = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.term_checker.reset()
        self._steps = 0

        # Generate obstacles
        self._obstacles = generate_random_obstacles(
            self.num_obstacles, self.bounds, rng=self.np_random
        )
        self._obstacle_positions = np.array([o.position for o in self._obstacles])
        self._obstacle_radii = np.array([o.radius for o in self._obstacles])

        # Initialize drone states (random positions in bounds, near ground)
        drone_init_positions = self._sample_drone_positions()
        self._states = np.zeros((self.num_drones, 12))
        for i in range(self.num_drones):
            self._states[i] = self.dynamics.reset_state(
                position=drone_init_positions[i],
                velocity=np.zeros(3),
                euler=np.zeros(3),
                angular_vel=np.zeros(3),
            )

        # Generate targets
        self._targets = generate_random_targets(
            self.num_targets,
            self.bounds,
            mode=self.task_mode,
            drone_positions=drone_init_positions,
            obstacle_positions=self._obstacle_positions,
            obstacle_radii=self._obstacle_radii,
            rng=self.np_random,
        )

        # Reset target states
        for t in self._targets:
            t.reached = False

        # Task assignment: greedy nearest-neighbor
        self._target_assignment = self._greedy_assignment(self._states, self._targets)
        self._target_assigned = np.zeros(self.num_targets, dtype=bool)
        self._carry_status = np.zeros(self.num_drones, dtype=bool)

        # Reset renderer-side accumulators
        self._last_rewards = np.zeros(self.num_drones, dtype=np.float32)
        self._prev_target_assigned = self._target_assigned.copy()
        self._prev_carry_status = self._carry_status.copy()
        self._render_event_queue = []
        self._rgb_buffer = []

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[
        Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]
    ]:
        """
        Args:
            action: (n_drones, 4) normalized continuous actions

        Returns:
            obs, reward, terminated, truncated, info
        """
        self._steps += 1

        # --- Snapshot state before dynamics for potential-based shaping ---
        prev_states = self._states.copy()

        # --- Apply dynamics ---
        raw_actions = np.array(
            [
                normalize_action(action[i], self.act_config)
                for i in range(self.num_drones)
            ]
        )

        for _ in range(self.aggregate_phy_steps):
            for i in range(self.num_drones):
                self._states[i] = self.dynamics.step(self._states[i], raw_actions[i])

        # NaN detection and position clamping
        for i in range(self.num_drones):
            # Fix NaN/Inf states
            if np.any(np.isnan(self._states[i])) or np.any(np.isinf(self._states[i])):
                self._states[i] = self._states[i].copy()
                self._states[i][np.isnan(self._states[i])] = 0.0
                self._states[i][np.isinf(self._states[i])] = 0.0
            # Clamp position to bounds (soft boundary, penalized via reward)
            self._states[i, 0] = np.clip(
                self._states[i, 0], self.bounds[0, 0], self.bounds[0, 1]
            )
            self._states[i, 1] = np.clip(
                self._states[i, 1], self.bounds[1, 0], self.bounds[1, 1]
            )
            self._states[i, 2] = np.clip(
                self._states[i, 2], self.bounds[2, 0], self.bounds[2, 1]
            )

        # --- Snapshot pre-update state for event detection ---
        prev_assigned = (
            self._target_assigned.copy() if self._target_assigned is not None else None
        )
        prev_carry = (
            self._carry_status.copy() if self._carry_status is not None else None
        )

        # --- Update target assignments & reached status ---
        self._update_target_status()

        # --- Detect render events (target_reached, pickup, delivery) ---
        if prev_assigned is not None and self._target_assigned is not None:
            newly_done = self._target_assigned & ~prev_assigned
            for j in np.where(newly_done)[0]:
                t = self._targets[int(j)]
                if t.target_type.value == 1:  # PICKUP
                    kind, color = "pickup", "lime"
                elif t.target_type.value == 2:  # DELIVERY
                    kind, color = "delivery", "gold"
                else:
                    kind, color = "target_reached", "deepskyblue"
                self._render_event_queue.append(
                    _make_event(kind, t.position, color=color, target_id=int(j))
                )

        # --- Check termination ---
        target_positions = np.array([t.position for t in self._targets])
        terminated, truncated, all_done = self.term_checker.check(
            self._states,
            target_positions,
            self._target_assigned,
            self._target_assignment,
            self._obstacle_positions,
            self._obstacle_radii,
            self._carry_status,
            self.bounds,
            self.task_mode,
        )

        # --- Detect collision events (heuristic) ---
        self._detect_collisions(target_positions)

        # --- Compute rewards ---
        reward_info = self.reward_calc.compute(
            self._states,
            prev_states,
            target_positions,
            self._target_assigned,
            self._target_assignment,
            self._obstacle_positions,
            self._obstacle_radii,
            raw_actions,
            self.bounds,
            all_done,
        )

        self._last_rewards = reward_info["agent_rewards"]

        # --- Build outputs ---
        obs = self._get_obs()
        info = self._get_info()
        info.update({"reward_components": reward_info["components"]})

        return obs, reward_info["agent_rewards"], terminated, truncated, info

    def render(self):
        """Render the environment according to ``self.render_mode``.

        Modes
        -----
        ``None``        - no-op, returns ``None``.
        ``"human"``     - update interactive matplotlib window, returns ``None``.
        ``"rgb_array"`` - return the current frame as an ``(H, W, 3)`` uint8 ndarray.
        ``"rgb_array_list"`` - append the current frame to an internal buffer and
            return the buffer. The buffer is reset on ``reset()`` and can be
            retrieved via :attr:`rgb_array_list`.
        ``"video"``     - alias of ``"rgb_array_list"`` (for clarity in user code).
        ``"top_down"``  - return a 2D top-down view (numpy array).
        """
        if self.render_mode is None:
            return None
        from utils.rendering import Matplotlib3DRenderer, RenderState, RenderEvent
        from utils.topdown import render_top_down  # local import to avoid cycles

        if self._renderer is None:
            show_top_down = self.render_mode != "top_down"
            self._renderer = Matplotlib3DRenderer(
                self.bounds,
                show_trail=True,
                show_attitude=True,
                show_hud=True,
                show_top_down=show_top_down,
            )

        info = self._get_info()
        info_payload = {
            "steps": self._steps,
            "reward": self._last_rewards,
            "completed": (
                int(np.sum(self._target_assigned))
                if self._target_assigned is not None
                else 0
            ),
            "total_targets": self.num_targets,
            "mode": self.task_mode,
        }
        state = RenderState(
            states=self._states,
            targets=self._targets,
            obstacles=self._obstacles,
            assignment=self._target_assignment,
            carry_status=self._carry_status,
            info=info_payload,
            events=self._render_event_queue,
        )
        self._renderer.update(state)
        # Clear the queue once consumed by the renderer
        self._render_event_queue = []

        if self.render_mode == "top_down":
            return render_top_down(self)

        if self.render_mode == "human":
            return None

        frame = self._renderer.get_frame()
        if frame is None:
            return None

        if self.render_mode in ("rgb_array_list", "video"):
            self._rgb_buffer.append(frame)
            return list(self._rgb_buffer)
        return frame

    @property
    def rgb_array_list(self) -> List[np.ndarray]:
        """Frames accumulated since the last ``reset()`` (read-only copy)."""
        return list(self._rgb_buffer)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._rgb_buffer = []

    # ---------------- Render helpers ----------------

    @staticmethod
    def _normalize_render_mode(mode):
        """Map deprecated / user-friendly mode names to canonical names."""
        if mode is None:
            return None
        if not isinstance(mode, str):
            raise ValueError(f"render_mode must be str or None, got {type(mode)}")
        canonical = {
            "human": "human",
            "rgb_array": "rgb_array",
            "rgb_array_list": "rgb_array_list",
            "rgb_list": "rgb_array_list",
            "video": "rgb_array_list",
            "top_down": "top_down",
            "ansi": "rgb_array",
        }
        if mode not in canonical:
            raise ValueError(
                f"Unknown render_mode={mode!r}. " f"Valid: {list(canonical.keys())}"
            )
        return canonical[mode]

    # ==================== Internal Methods ====================

    def _sample_drone_positions(self) -> np.ndarray:
        """Sample initial drone positions within bounds, away from obstacles."""
        positions = np.zeros((self.num_drones, 3))
        for i in range(self.num_drones):
            valid = False
            for _ in range(1000):
                pos = self.np_random.uniform(
                    [self.bounds[0, 0], self.bounds[1, 0], 1.0],
                    [self.bounds[0, 1], self.bounds[1, 1], 5.0],
                )
                # Check distance to other drones
                if i > 0 and np.any(
                    np.linalg.norm(positions[:i, :2] - pos[:2], axis=1) < 5.0
                ):
                    continue
                # Check distance to obstacles
                if len(self._obstacle_positions) > 0:
                    dists = np.linalg.norm(
                        self._obstacle_positions[:, :2] - pos[:2], axis=1
                    )
                    if np.any(dists < self._obstacle_radii + 3.0):
                        continue
                valid = True
                break
            if not valid:
                pos = np.array([self.bounds[0, 0] + i * 5.0, self.bounds[1, 0], 1.0])
            positions[i] = pos
        return positions

    def _greedy_assignment(
        self, states: np.ndarray, targets: List[Target]
    ) -> np.ndarray:
        """Simple nearest-neighbor assignment for initialization."""
        n_drones = len(states)
        n_targets = len(targets)
        assignment = np.full(n_drones, -1, dtype=int)

        if n_targets == 0:
            return assignment

        target_pos = np.array([t.position for t in targets])
        available_targets = set(range(n_targets))
        drones = list(range(n_drones))

        # Sort drones by some heuristic (closest to any target)
        min_dists = np.array(
            [
                np.min(np.linalg.norm(target_pos - states[i, :3], axis=1))
                for i in range(n_drones)
            ]
        )
        drone_order = np.argsort(min_dists)

        for i in drone_order:
            if not available_targets:
                break
            pos_i = states[i, :3]
            dists = {
                t: np.linalg.norm(target_pos[t] - pos_i) for t in available_targets
            }
            best_target = min(dists, key=dists.get)
            assignment[i] = best_target
            available_targets.remove(best_target)

        return assignment

    def _update_target_status(self):
        """Check which targets have been reached and update assignment."""
        if self.task_mode == "reach":
            for i in range(self.num_drones):
                aid = self._target_assignment[i]
                if aid >= 0 and not self._targets[aid].reached:
                    dist = np.linalg.norm(
                        self._states[i, :3] - self._targets[aid].position
                    )
                    if dist < self._targets[aid].radius:
                        self._targets[aid].reached = True
                        self._target_assigned[aid] = True

        elif self.task_mode == "delivery":
            for i in range(self.num_drones):
                aid = self._target_assignment[i]
                if aid < 0:
                    continue
                t = self._targets[aid]
                dist = np.linalg.norm(self._states[i, :3] - t.position)
                if dist < t.radius:
                    if t.target_type == TargetType.PICKUP and not t.reached:
                        t.reached = True
                        self._carry_status[i] = True
                        # Find paired delivery target
                        delivery_idx = self._find_paired_delivery(t.pair_id)
                        if delivery_idx >= 0:
                            self._target_assignment[i] = delivery_idx
                    elif t.target_type == TargetType.DELIVERY and self._carry_status[i]:
                        t.reached = True
                        self._carry_status[i] = False
                        self._target_assigned[aid] = True

    def _find_paired_delivery(self, pair_id: int) -> int:
        """Find the delivery target with the given pair_id."""
        for i, t in enumerate(self._targets):
            if t.target_type == TargetType.DELIVERY and t.pair_id == pair_id:
                return i
        return -1

    def _detect_collisions(self, target_positions: np.ndarray) -> None:
        """Detect obstacle and drone-drone collisions; enqueue render events.

        Uses the same ``Obstacle.distance_to`` signed distance used by
        :class:`core.termination.TerminationChecker` so events stay consistent
        with collision-based termination.
        """
        # Drone vs obstacle
        for i in range(self.num_drones):
            pos = self._states[i, :3]
            for k, obs in enumerate(self._obstacles):
                if obs.distance_to(pos) < 0.0:
                    self._render_event_queue.append(
                        _make_event(
                            "collision", pos, color="red", drone_id=i, duration=0.8
                        )
                    )
        # Drone vs drone
        if self.num_drones >= 2:
            for i in range(self.num_drones):
                for j in range(i + 1, self.num_drones):
                    d = float(np.linalg.norm(self._states[i, :3] - self._states[j, :3]))
                    if d < 1.0:  # close-encounter threshold
                        midpoint = 0.5 * (self._states[i, :3] + self._states[j, :3])
                        self._render_event_queue.append(
                            _make_event(
                                "collision",
                                midpoint,
                                color="orange",
                                drone_id=i,
                                duration=0.8,
                            )
                        )

    def _get_obs(self) -> Dict[str, np.ndarray]:
        target_positions = np.array([t.position for t in self._targets])
        agent_obs = np.zeros((self.num_drones, self.agent_obs_dim), dtype=np.float32)
        for i in range(self.num_drones):
            agent_obs[i] = build_local_obs(
                i,
                self._states,
                target_positions,
                self._target_assigned,
                self._obstacle_positions,
                self._obstacle_radii,
                self._carry_status,
                self.obs_config,
            )
        global_obs = build_global_obs(
            self._states,
            target_positions,
            self._target_assigned,
            self._obstacle_positions,
            self._obstacle_radii,
            self._carry_status,
            self.obs_config,
        )
        return {"agent_obs": agent_obs, "global_obs": global_obs}

    def _get_info(self) -> Dict[str, Any]:
        target_positions = np.array([t.position for t in self._targets])
        return {
            "states": self._states.copy(),
            "target_positions": target_positions,
            "target_assigned": self._target_assigned.copy(),
            "target_assignment": self._target_assignment.copy(),
            "obstacle_positions": self._obstacle_positions.copy(),
            "obstacle_radii": self._obstacle_radii.copy(),
            "carry_status": self._carry_status.copy(),
            "steps": self._steps,
        }

    def _global_obs_dim(self) -> int:
        return (
            self.num_drones * 12
            + self.num_targets * 3
            + self.num_targets
            + self.num_obstacles * 3
            + self.num_obstacles
            + self.num_drones
        )

    # ==================== Property accessors ====================

    @property
    def all_states(self) -> np.ndarray:
        return self._states

    @property
    def targets(self) -> List[Target]:
        return self._targets

    @property
    def obstacles(self) -> List[Obstacle]:
        return self._obstacles
