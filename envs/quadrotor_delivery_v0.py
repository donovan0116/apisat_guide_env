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

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.dynamics import QuadrotorDynamics, QuadrotorParams
from core.state import ObsConfig, build_local_obs, build_global_obs
from core.action import ActionConfig, normalize_action
from core.target import (
    Target, TargetType, Obstacle,
    generate_random_targets, generate_random_obstacles,
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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        num_drones: int = 4,
        num_targets: int = 4,
        num_obstacles: int = 10,
        task_mode: str = "reach",
        bounds: np.ndarray = None,       # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
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
        self.bounds = bounds if bounds is not None else np.array([
            [-50.0, 50.0], [-50.0, 50.0], [0.0, 30.0]
        ])
        self.aggregate_phy_steps = aggregate_phy_steps
        self.render_mode = render_mode
        self.device = device

        # --- Sub-modules ---
        self.dynamics = QuadrotorDynamics(quad_params)
        self.obs_config = obs_config or ObsConfig(
            max_num_drones=num_drones, max_num_targets=num_targets,
            max_num_obstacles=num_obstacles,
        )
        self.act_config = act_config or ActionConfig()
        self.reward_calc = RewardCalculator(reward_config)
        self.term_checker = TerminationChecker(term_config)

        # --- Spaces ---
        self.agent_obs_dim = self.obs_config.total_dim
        self.agent_act_dim = self.dynamics.action_dim

        self.observation_space = spaces.Dict({
            "agent_obs": spaces.Box(
                -1.0, 1.0, shape=(num_drones, self.agent_obs_dim), dtype=np.float32
            ),
            "global_obs": spaces.Box(
                -np.inf, np.inf,
                shape=(self._global_obs_dim(),), dtype=np.float32
            ),
        })
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(num_drones, self.agent_act_dim), dtype=np.float32
        )

        # --- Internal state ---
        self._states: np.ndarray = None        # [n_drones, 12]
        self._targets: List[Target] = None
        self._obstacles: List[Obstacle] = None
        self._obstacle_positions: np.ndarray = None
        self._obstacle_radii: np.ndarray = None
        self._carry_status: np.ndarray = None     # [n_drones] bool
        self._target_assigned: np.ndarray = None  # [n_targets] bool
        self._target_assignment: np.ndarray = None  # [n_drones] int
        self._steps: int = 0

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
            self.num_targets, self.bounds, mode=self.task_mode,
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

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[
        Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]
    ]:
        """
        Args:
            action: (n_drones, 4) normalized continuous actions

        Returns:
            obs, reward, terminated, truncated, info
        """
        self._steps += 1

        # --- Apply dynamics ---
        raw_actions = np.array([
            normalize_action(action[i], self.act_config)
            for i in range(self.num_drones)
        ])

        for _ in range(self.aggregate_phy_steps):
            for i in range(self.num_drones):
                self._states[i] = self.dynamics.step(self._states[i], raw_actions[i])

        # NaN detection: mark drones with NaN state as terminated
        for i in range(self.num_drones):
            if np.any(np.isnan(self._states[i])) or np.any(np.isinf(self._states[i])):
                self._states[i] = self._states[i].copy()
                self._states[i][np.isnan(self._states[i])] = 0.0
                self._states[i][np.isinf(self._states[i])] = 0.0

        # --- Update target assignments & reached status ---
        self._update_target_status()

        # --- Check termination ---
        target_positions = np.array([t.position for t in self._targets])
        terminated, truncated, all_done = self.term_checker.check(
            self._states, target_positions,
            self._target_assigned, self._target_assignment,
            self._obstacle_positions, self._obstacle_radii,
            self._carry_status, self.bounds, self.task_mode,
        )

        # --- Compute rewards ---
        reward_info = self.reward_calc.compute(
            self._states, target_positions,
            self._target_assigned, self._target_assignment,
            self._obstacle_positions, self._obstacle_radii,
            raw_actions, self.bounds, all_done,
        )

        # --- Build outputs ---
        obs = self._get_obs()
        info = self._get_info()
        info.update({"reward_components": reward_info["components"]})

        return obs, reward_info["agent_rewards"], terminated, truncated, info

    def render(self):
        """Render the environment (placeholder, see rendering.py for full impl)."""
        from utils.rendering import SimpleRenderer
        if not hasattr(self, "_renderer"):
            self._renderer = SimpleRenderer(self.bounds)
        return self._renderer.render(
            self._states, self._targets, self._obstacles,
            self._target_assignment, mode=self.render_mode,
        )

    def close(self):
        if hasattr(self, "_renderer"):
            self._renderer.close()

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
                if i > 0 and np.any(np.linalg.norm(positions[:i, :2] - pos[:2], axis=1) < 5.0):
                    continue
                # Check distance to obstacles
                if len(self._obstacle_positions) > 0:
                    dists = np.linalg.norm(self._obstacle_positions[:, :2] - pos[:2], axis=1)
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
        min_dists = np.array([
            np.min(np.linalg.norm(target_pos - states[i, :3], axis=1))
            for i in range(n_drones)
        ])
        drone_order = np.argsort(min_dists)

        for i in drone_order:
            if not available_targets:
                break
            pos_i = states[i, :3]
            dists = {t: np.linalg.norm(target_pos[t] - pos_i) for t in available_targets}
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

    def _get_obs(self) -> Dict[str, np.ndarray]:
        target_positions = np.array([t.position for t in self._targets])
        agent_obs = np.zeros((self.num_drones, self.agent_obs_dim), dtype=np.float32)
        for i in range(self.num_drones):
            agent_obs[i] = build_local_obs(
                i, self._states, target_positions,
                self._target_assigned, self._obstacle_positions,
                self._obstacle_radii, self._carry_status, self.obs_config,
            )
        global_obs = build_global_obs(
            self._states, target_positions, self._target_assigned,
            self._obstacle_positions, self._obstacle_radii,
            self._carry_status, self.obs_config,
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
        return (self.num_drones * 12 + self.num_targets * 3 +
                self.num_targets + self.num_obstacles * 3 +
                self.num_obstacles + self.num_drones)

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
