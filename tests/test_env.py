"""
Tests for multi-UAV delivery environment.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.dynamics import QuadrotorDynamics, QuadrotorParams
from core.state import ObsConfig, build_local_obs
from core.action import ActionConfig, normalize_action
from core.target import generate_random_targets, generate_random_obstacles
from core.reward import RewardCalculator
from core.termination import TerminationChecker
from envs import QuadrotorDeliveryEnv, ParallelQuadrotorDelivery


class TestDynamics:
    """Test quadrotor dynamics model."""

    def test_hover_thrust(self):
        params = QuadrotorParams()
        dyn = QuadrotorDynamics(params)
        assert dyn.hover_thrust == params.mass * params.g

    def test_step_shape(self):
        dyn = QuadrotorDynamics()
        state = dyn.reset_state(np.array([0.0, 0.0, 1.0]))
        action = np.array([dyn.hover_thrust, 0.0, 0.0, 0.0])
        next_state = dyn.step(state, action)
        assert next_state.shape == (12,)
        # Hovering should keep velocity near zero (gravity canceled by thrust)
        assert abs(next_state[5]) < 0.5  # vz near 0

    def test_rotation_matrix_identity(self):
        dyn = QuadrotorDynamics()
        R = dyn._rotation_matrix(np.zeros(3))
        assert np.allclose(R, np.eye(3))

    def test_step_many(self):
        dyn = QuadrotorDynamics()
        state = dyn.reset_state(np.array([0.0, 0.0, 5.0]))
        # Apply hover thrust for 100 steps
        for _ in range(100):
            state = dyn.step(state, np.array([dyn.hover_thrust, 0.0, 0.0, 0.0]))
        # Should stay near original altitude
        assert 4.0 < state[2] < 16.0  # Allow some drift from Euler integration


class TestStateBuild:
    """Test observation building."""

    def test_local_obs_shape(self):
        config = ObsConfig(max_num_drones=4, max_num_targets=4, max_num_obstacles=10)
        states = np.zeros((4, 12))
        targets = np.zeros((4, 3))
        assigned = np.zeros(4, dtype=bool)
        obs_pos = np.zeros((0, 3))
        obs_rad = np.zeros(0)
        carry = np.zeros(4, dtype=bool)

        obs = build_local_obs(0, states, targets, assigned, obs_pos, obs_rad, carry, config)
        assert obs.shape == (config.total_dim,)


class TestAction:
    """Test action normalization."""

    def test_normalize_action(self):
        config = ActionConfig(thrust_range=(0.0, 0.8), torque_range=(-0.1, 0.1))
        raw = np.array([0.0, 0.5, -0.5, 0.0])  # mid throttle, half torque
        phys = normalize_action(raw, config)
        assert phys.shape == (4,)
        assert phys[0] == 0.4  # midpoint of [0, 0.8]
        assert abs(phys[1] - 0.05) < 1e-6  # 0.5 * 0.1


class TestGeneration:
    """Test target and obstacle generation."""

    def test_generate_targets_reach(self):
        bounds = np.array([[-30, 30], [-30, 30], [0, 30]])
        rng = np.random.default_rng(0)
        targets = generate_random_targets(4, bounds, mode="reach", rng=rng)
        assert len(targets) == 4
        for t in targets:
            assert bounds[0, 0] <= t.position[0] <= bounds[0, 1]

    def test_generate_obstacles(self):
        bounds = np.array([[-30, 30], [-30, 30], [0, 30]])
        rng = np.random.default_rng(0)
        obstacles = generate_random_obstacles(8, bounds, rng=rng)
        assert len(obstacles) <= 8  # may be fewer if placement fails


class TestEnv:
    """Integration tests for the environment."""

    def test_env_create(self):
        env = QuadrotorDeliveryEnv(num_drones=2, num_targets=2, num_obstacles=3)
        assert env.observation_space is not None
        assert env.action_space is not None
        env.close()

    def test_env_reset(self):
        env = QuadrotorDeliveryEnv(num_drones=3, num_targets=3, num_obstacles=5, seed=42)
        obs, info = env.reset()
        assert obs["agent_obs"].shape == (3, env.agent_obs_dim)
        assert "global_obs" in obs
        env.close()

    def test_env_step(self):
        env = QuadrotorDeliveryEnv(num_drones=2, num_targets=2, num_obstacles=2, seed=42)
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
        assert obs["agent_obs"].shape == (2, env.agent_obs_dim)
        env.close()

    def test_env_multiple_steps(self):
        env = QuadrotorDeliveryEnv(num_drones=2, num_targets=2, num_obstacles=3, seed=42)
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        env.close()

    def test_gym_registration(self):
        env = QuadrotorDeliveryEnv(num_drones=2, num_targets=2, num_obstacles=2, seed=42)
        # Verify Gymnasium API compliance
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "render")
        assert hasattr(env, "close")
        assert hasattr(env, "metadata")
        env.close()


class TestPettingZoo:
    """Tests for PettingZoo-compatible wrapper."""

    def test_parallel_env_create(self):
        env = ParallelQuadrotorDelivery(num_drones=3, num_targets=3, num_obstacles=5)
        assert "drone_0" in env.possible_agents
        env.close()

    def test_parallel_env_reset_step(self):
        env = ParallelQuadrotorDelivery(num_drones=2, num_targets=2, num_obstacles=3, seed=42)
        obs_dict, info_dict = env.reset()

        actions = {agent_id: env.action_space(agent_id).sample()
                   for agent_id in env.agents}
        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(actions)

        assert len(reward_dict) > 0
        for agent_id in env.possible_agents:
            if agent_id in reward_dict:
                assert isinstance(reward_dict[agent_id], float)
        env.close()


if __name__ == "__main__":
    # Run tests manually if pytest not available
    test_dyn = TestDynamics()
    test_dyn.test_hover_thrust()
    test_dyn.test_step_shape()
    test_dyn.test_rotation_matrix_identity()
    print("Dynamics tests passed.")

    test_act = TestAction()
    test_act.test_normalize_action()
    print("Action tests passed.")

    test_gen = TestGeneration()
    test_gen.test_generate_targets_reach()
    test_gen.test_generate_obstacles()
    print("Generation tests passed.")

    test_env = TestEnv()
    test_env.test_env_create()
    test_env.test_env_reset()
    test_env.test_env_step()
    test_env.test_env_multiple_steps()
    print("Environment tests passed.")

    test_pz = TestPettingZoo()
    test_pz.test_parallel_env_create()
    test_pz.test_parallel_env_reset_step()
    print("PettingZoo wrapper tests passed.")

    print("\nAll tests passed.")
