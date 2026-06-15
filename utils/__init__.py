from .geometry import distance_point_to_line, check_line_obstacle_collision, wrap_angle
from .config import FullConfig, EnvConfig, TrainingConfig, get_default_config

_RENDERING_EXPORTS = {
    "Renderer",
    "Matplotlib3DRenderer",
    "SimpleRenderer",
    "RenderState",
    "RenderEvent",
}


def __getattr__(name):
    if name in _RENDERING_EXPORTS:
        from . import rendering
        return getattr(rendering, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    from .visualization import play_episode, record_episode, live_render
except ImportError:  # pragma: no cover - matplotlib might be missing
    pass

try:
    from .topdown import render_top_down
except ImportError:  # pragma: no cover
    pass


__all__ = [
    "distance_point_to_line",
    "check_line_obstacle_collision",
    "wrap_angle",
    "FullConfig",
    "EnvConfig",
    "TrainingConfig",
    "get_default_config",
    *_RENDERING_EXPORTS,
]
