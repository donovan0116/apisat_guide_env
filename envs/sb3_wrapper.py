"""
Wrappers for Stable-Baselines3 compatibility.

- FlattenDictObs: Dict obs -> flat Box
- FlattenAction: (n_drones, act_dim) -> (n_drones*act_dim,)
- AggregateMARL: per-agent reward/terminated/truncated -> scalar
- make_sb3_env: apply all wrappers for centralized training
"""

from typing import Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlattenDictObs(gym.ObservationWrapper):
    """
    Flattens the Dict observation into a single Box array.

    Input obs: {"agent_obs": (n, d1), "global_obs": (d2,)}
    Output obs: (n * d1 + d2,)

    For centralized PPO where a single policy controls all agents.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        orig = env.observation_space
        assert isinstance(orig, spaces.Dict), "Env must have Dict observation space"

        agent_dim = orig["agent_obs"].shape[0] * orig["agent_obs"].shape[1]
        global_dim = orig["global_obs"].shape[0]
        total_dim = agent_dim + global_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )
        self._agent_shape = orig["agent_obs"].shape
        self._global_shape = orig["global_obs"].shape

    def observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        agent_flat = obs["agent_obs"].flatten()
        global_flat = obs["global_obs"].flatten()
        return np.concatenate([agent_flat, global_flat]).astype(np.float32)


class FlattenAction(gym.ActionWrapper):
    """
    Ensures action has correct shape for SB3 (n_agents * act_dim,).

    SB3 expects action_space shape to be flat.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        orig = env.action_space

        if isinstance(orig, spaces.Box) and len(orig.shape) > 1:
            # (n_drones, act_dim) -> (n_drones * act_dim,)
            flat_dim = int(np.prod(orig.shape))
            self.action_space = spaces.Box(
                low=orig.low.flatten()[0],
                high=orig.high.flatten()[0],
                shape=(flat_dim,),
                dtype=np.float32,
            )
            self._orig_shape = orig.shape
            self._rewrap_action = True
        else:
            self._rewrap_action = False

    def action(self, action: np.ndarray) -> np.ndarray:
        if self._rewrap_action:
            return action.reshape(self._orig_shape)
        return action


class AggregateMARL(gym.Wrapper):
    """
    Aggregates per-agent reward/terminated/truncated into scalars.

    For centralized PPO: sum rewards, any-terminated, any-truncated.
    """

    def __init__(self, env: gym.Env, reward_agg: str = "sum"):
        super().__init__(env)
        self.reward_agg = reward_agg

    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step(action)

        if isinstance(rewards, np.ndarray) and rewards.ndim > 0:
            if self.reward_agg == "sum":
                reward = float(np.sum(rewards))
            elif self.reward_agg == "mean":
                reward = float(np.mean(rewards))
            else:
                reward = float(rewards[0])
        else:
            reward = float(rewards)

        if isinstance(terminated, np.ndarray):
            term = bool(np.any(terminated))
        else:
            term = bool(terminated)

        if isinstance(truncated, np.ndarray):
            trunc = bool(np.any(truncated))
        else:
            trunc = bool(truncated)

        info["per_agent_rewards"] = rewards
        info["per_agent_terminated"] = terminated
        info["per_agent_truncated"] = truncated

        return obs, reward, term, trunc, info


def make_sb3_env(env: gym.Env) -> gym.Env:
    """Wrap environment for Stable-Baselines3 centralized training."""
    env = FlattenDictObs(env)
    env = FlattenAction(env)
    env = AggregateMARL(env)
    return env


def make_sb3_single_env(env: gym.Env) -> gym.Env:
    """
    Wrap for single-agent SB3 training where each call
    returns a flat obs and scalar reward/terminated/truncated.
    """
    env = FlattenDictObs(env)
    env = FlattenAction(env)
    env = AggregateMARL(env)
    return env
