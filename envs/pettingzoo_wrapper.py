"""
PettingZoo Parallel API wrapper for multi-agent training.

Provides per-agent observations, actions, rewards, and done flags
while using the centralized QuadrotorDeliveryEnv underneath.
"""

from typing import Dict, Optional, Any
import functools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.quadrotor_delivery_v0 import QuadrotorDeliveryEnv
from core.state import ObsConfig, build_local_obs


class ParallelQuadrotorDelivery:
    """
    PettingZoo-compatible parallel environment wrapper.

    Conforms to the ParallelEnv API:
        - reset() -> (observations, infos)
        - step(actions) -> (observations, rewards, terminations, truncations, infos)
        - observation_space(agent) -> Space
        - action_space(agent) -> Space
        - agents -> list of agent ids
    """

    def __init__(self, **env_kwargs):
        self._env = QuadrotorDeliveryEnv(**env_kwargs)
        self._num_drones = self._env.num_drones
        self.possible_agents = [f"drone_{i}" for i in range(self._num_drones)]
        self.agents = self.possible_agents[:]
        self._agent_obs_dim = self._env.agent_obs_dim
        self._agent_act_dim = self._env.agent_act_dim

    @functools.lru_cache(maxsize=1)
    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(self._agent_obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=1)
    def action_space(self, agent: str) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(self._agent_act_dim,), dtype=np.float32)

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        full_obs, full_info = self._env.reset(seed=seed, options=options)

        obs_dict = {}
        info_dict = {}
        for i, agent_id in enumerate(self.possible_agents):
            obs_dict[agent_id] = full_obs["agent_obs"][i]
            info_dict[agent_id] = {
                "state": self._env._states[i].copy(),
                "assignment": int(self._env._target_assignment[i]),
            }
        # Add global obs for centralized training
        info_dict["global_obs"] = full_obs["global_obs"]

        self.agents = self.possible_agents[:]
        return obs_dict, info_dict

    def step(self, actions: Dict[str, np.ndarray]) -> tuple:
        # Convert per-agent actions to flat array
        action_array = np.zeros((self._num_drones, self._agent_act_dim), dtype=np.float32)
        for i, agent_id in enumerate(self.possible_agents):
            action_array[i] = actions[agent_id]

        full_obs, rewards, terminated, truncated, full_info = self._env.step(action_array)

        obs_dict = {}
        reward_dict = {}
        term_dict = {}
        trunc_dict = {}
        info_dict = {}

        for i, agent_id in enumerate(self.possible_agents):
            obs_dict[agent_id] = full_obs["agent_obs"][i]
            reward_dict[agent_id] = float(rewards[i])
            term_dict[agent_id] = bool(terminated[i])
            trunc_dict[agent_id] = bool(truncated[i])
            info_dict[agent_id] = {
                "state": self._env._states[i].copy(),
                "assignment": int(self._env._target_assignment[i]),
                "reward_components": {
                    k: float(v[i]) for k, v in full_info.get("reward_components", {}).items()
                },
            }

        # Global obs
        info_dict["global_obs"] = full_obs["global_obs"]

        # Remove terminated agents
        for i, agent_id in enumerate(self.possible_agents):
            if terminated[i] or truncated[i]:
                if agent_id in self.agents:
                    self.agents.remove(agent_id)

        # Check all done
        if not self.agents:
            self.agents = []

        return obs_dict, reward_dict, term_dict, trunc_dict, info_dict

    def render(self) -> Optional[np.ndarray]:
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def env(self) -> QuadrotorDeliveryEnv:
        return self._env
