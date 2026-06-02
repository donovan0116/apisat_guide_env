"""
Tests for the layered rendering system.

Validates:
  * ``Matplotlib3DRenderer`` produces correctly shaped RGB frames.
  * Artist pool is reused across updates (figure is not recreated).
  * Trails accumulate per-drone position history.
  * ``record_episode`` and ``play_episode`` write valid video files.
  * All new render modes in :class:`QuadrotorDeliveryEnv` work as documented.
"""

import os
import sys
import tempfile
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force a non-interactive backend for the test session
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)

from core.dynamics import QuadrotorDynamics
from core.target import Obstacle, Target, TargetType
from envs import QuadrotorDeliveryEnv
from utils.rendering import (
    Matplotlib3DRenderer,
    RenderEvent,
    RenderState,
    SimpleRenderer,
)


def _make_scene(n_drones: int = 3, n_targets: int = 3, n_obstacles: int = 2):
    rng = np.random.default_rng(0)
    bounds = np.array([[-30, 30], [-30, 30], [0, 30]])
    obstacles = [
        Obstacle(position=rng.uniform(bounds[:, 0], bounds[:, 1]),
                 shape="cylinder",
                 size=np.array([2.0, 0.0, 6.0]))
        for _ in range(n_obstacles)
    ]
    targets = [
        Target(position=rng.uniform(bounds[:, 0], bounds[:, 1]),
               target_type=TargetType.REACH)
        for _ in range(n_targets)
    ]
    dyn = QuadrotorDynamics()
    states = np.zeros((n_drones, 12))
    for i in range(n_drones):
        states[i] = dyn.reset_state(
            position=rng.uniform(bounds[:, 0], bounds[:, 1]),
            euler=np.array([0.1, 0.1, 0.0]),
        )
    assignment = np.array(list(range(min(n_drones, n_targets))) +
                          [-1] * max(0, n_drones - n_targets))
    return states, targets, obstacles, assignment, bounds


# ---------------------------------------------------------------------------
# Matplotlib3DRenderer
# ---------------------------------------------------------------------------


class TestMatplotlib3DRenderer:
    def test_get_frame_shape(self):
        states, targets, obstacles, assignment, bounds = _make_scene()
        r = Matplotlib3DRenderer(bounds)
        r.update(RenderState(
            states=states, targets=targets, obstacles=obstacles,
            assignment=assignment,
            carry_status=np.zeros(states.shape[0], dtype=bool),
        ))
        frame = r.get_frame()
        assert frame is not None
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        assert frame.shape[0] > 0 and frame.shape[1] > 0
        r.close()

    def test_artist_pool_reuse(self):
        """Updating the renderer must NOT recreate the figure or axes."""
        states, targets, obstacles, assignment, bounds = _make_scene()
        r = Matplotlib3DRenderer(bounds)
        state = RenderState(
            states=states, targets=targets, obstacles=obstacles,
            assignment=assignment,
            carry_status=np.zeros(states.shape[0], dtype=bool),
        )
        r.update(state)
        fig1 = r._fig
        ax1 = r._ax
        r.update(state)
        fig2 = r._fig
        ax2 = r._ax
        assert fig1 is fig2
        assert ax1 is ax2
        r.close()

    def test_trail_accumulation(self):
        states, targets, obstacles, assignment, bounds = _make_scene(
            n_drones=2, n_targets=2, n_obstacles=1,
        )
        r = Matplotlib3DRenderer(bounds, trail_length=50, show_trail=True)
        n_steps = 12
        for step in range(n_steps):
            new_states = states.copy()
            new_states[:, 2] += step * 0.1
            r.update(RenderState(
                states=new_states, targets=targets, obstacles=obstacles,
                assignment=assignment,
                carry_status=np.zeros(new_states.shape[0], dtype=bool),
            ))
        assert len(r._trails) == 2
        for tr in r._trails:
            assert len(tr) == n_steps
        r.close()

    def test_trail_capped(self):
        states, targets, obstacles, assignment, bounds = _make_scene(n_drones=1)
        r = Matplotlib3DRenderer(bounds, trail_length=5, show_trail=True)
        for step in range(20):
            new = states.copy()
            new[:, 2] += step * 0.1
            r.update(RenderState(
                states=new, targets=targets, obstacles=obstacles,
                assignment=assignment, carry_status=np.array([False]),
            ))
        assert len(r._trails[0]) == 5  # capped
        r.close()

    def test_event_pulse(self):
        states, targets, obstacles, assignment, bounds = _make_scene()
        r = Matplotlib3DRenderer(bounds)
        r.update(RenderState(
            states=states, targets=targets, obstacles=obstacles,
            assignment=assignment,
            carry_status=np.zeros(states.shape[0], dtype=bool),
            events=[RenderEvent(
                kind="target_reached", position=targets[0].position,
                color="lime",
            )],
        ))
        halos = r._active_halo_points()
        assert halos is not None
        assert len(halos) >= 1
        r.close()

    def test_close_releases_figure(self):
        r = Matplotlib3DRenderer(np.array([[-30, 30], [-30, 30], [0, 30]]))
        states, targets, obstacles, assignment, _ = _make_scene(n_drones=2)
        r.update(RenderState(
            states=states, targets=targets, obstacles=obstacles,
            assignment=assignment,
            carry_status=np.zeros(states.shape[0], dtype=bool),
        ))
        r.close()
        assert r._fig is None
        assert r._ax is None
        assert r._artists is None


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestSimpleRendererAlias:
    def test_simple_renderer_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sr = SimpleRenderer(np.array([[-10, 10], [-10, 10], [0, 10]]))
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        assert isinstance(sr, Matplotlib3DRenderer)
        sr.close()

    def test_simple_renderer_rgb_array(self):
        states, targets, obstacles, assignment, _ = _make_scene()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sr = SimpleRenderer(np.array([[-30, 30], [-30, 30], [0, 30]]))
        frame = sr.render(states, targets, obstacles, assignment,
                          mode="rgb_array")
        assert frame is not None
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        sr.close()

    def test_simple_renderer_human(self):
        states, targets, obstacles, assignment, _ = _make_scene()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sr = SimpleRenderer(np.array([[-30, 30], [-30, 30], [0, 30]]))
        out = sr.render(states, targets, obstacles, assignment, mode="human")
        assert out is None
        sr.close()


# ---------------------------------------------------------------------------
# Env integration
# ---------------------------------------------------------------------------


class TestEnvRenderModes:
    def _make_env(self, render_mode):
        return QuadrotorDeliveryEnv(
            num_drones=2, num_targets=2, num_obstacles=2,
            render_mode=render_mode, seed=42,
        )

    def test_rgb_array(self):
        env = self._make_env("rgb_array")
        env.reset()
        env.step(env.action_space.sample())
        frame = env.render()
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        env.close()

    def test_rgb_array_list(self):
        env = self._make_env("rgb_array_list")
        env.reset()
        for _ in range(4):
            env.step(env.action_space.sample())
            env.render()
        assert len(env.rgb_array_list) == 4
        env.close()

    def test_top_down(self):
        env = self._make_env("top_down")
        env.reset()
        env.step(env.action_space.sample())
        frame = env.render()
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        env.close()

    def test_video_mode_alias(self):
        env = self._make_env("video")
        env.reset()
        env.step(env.action_space.sample())
        out = env.render()
        assert isinstance(out, list) and len(out) == 1
        env.close()

    def test_none_mode(self):
        env = self._make_env(None)
        env.reset()
        env.step(env.action_space.sample())
        assert env.render() is None
        env.close()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            self._make_env("bogus_mode")

    def test_metadata_fps(self):
        env = self._make_env(None)
        assert "render_fps" in env.metadata
        assert env.metadata["render_fps"] == 30
        assert "human" in env.metadata["render_modes"]
        assert "rgb_array" in env.metadata["render_modes"]
        assert "rgb_array_list" in env.metadata["render_modes"]
        env.close()

    def test_reset_clears_rgb_buffer(self):
        env = self._make_env("rgb_array_list")
        env.reset()
        for _ in range(3):
            env.step(env.action_space.sample())
            env.render()
        assert len(env.rgb_array_list) == 3
        env.reset()
        assert len(env.rgb_array_list) == 0
        env.close()


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------


class TestRecordEpisode:
    def test_record_mp4(self):
        from utils.visualization import record_episode
        env = self._env_for_recording("rgb_array")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.mp4")
            result_path = record_episode(
                env, path, fps=8, max_steps=5, dpi=60,
            )
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 0
            # Verify it parses as a valid video file
            import imageio.v3 as iio
            meta = iio.immeta(result_path)
            assert meta.get("fps", 0) > 0
            assert meta.get("plugin") == "ffmpeg"
        env.close()

    def test_record_gif(self):
        from utils.visualization import record_episode
        env = self._env_for_recording("rgb_array")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gif")
            result_path = record_episode(
                env, path, fps=8, max_steps=5, dpi=60,
            )
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 0
        env.close()

    def _env_for_recording(self, render_mode):
        return QuadrotorDeliveryEnv(
            num_drones=2, num_targets=2, num_obstacles=2,
            render_mode=render_mode, seed=42,
        )


class TestPlayEpisode:
    def test_play_episode_save(self):
        from utils.visualization import play_episode
        env = QuadrotorDeliveryEnv(
            num_drones=2, num_targets=2, num_obstacles=2, seed=42,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "play.gif")
            info = play_episode(env, fps=8, max_steps=6, save_path=path,
                                dpi=60)
            assert info["frames"] > 0
            assert info["duration_sec"] > 0
            assert info["save_path"] == path
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        env.close()


# ---------------------------------------------------------------------------
# Top-down renderer
# ---------------------------------------------------------------------------


class TestTopDownRenderer:
    def test_top_down_shape(self):
        from utils.topdown import render_top_down
        env = QuadrotorDeliveryEnv(
            num_drones=2, num_targets=2, num_obstacles=2, seed=42,
        )
        env.reset()
        env.step(env.action_space.sample())
        frame = render_top_down(env)
        assert frame is not None
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        env.close()

    def test_top_down_trails_grow(self):
        from utils.topdown import render_top_down
        env = QuadrotorDeliveryEnv(
            num_drones=2, num_targets=2, num_obstacles=2, seed=42,
        )
        env.reset()
        env._topdown_trails = [__import__("collections").deque(maxlen=20)
                                for _ in range(env.num_drones)]
        for _ in range(5):
            env.step(env.action_space.sample())
            render_top_down(env)
        for tr in env._topdown_trails:
            assert len(tr) == 5
        env.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
