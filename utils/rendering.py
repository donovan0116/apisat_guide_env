"""
Layered 3D renderer for the multi-UAV delivery environment.

Architecture
------------
- ``RenderState`` / ``RenderEvent`` capture a single frame's data.
- ``Renderer`` is the abstract base class following the Gymnasium 1.0
  render protocol (``update`` / ``get_frame`` / ``close``).
- ``Matplotlib3DRenderer`` is the full implementation: artist pooling,
  trails, attitude indicators, HUD, 2D top-down inset and event flashes.
- ``SimpleRenderer`` is preserved as a deprecated alias for backward
  compatibility with the original API.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection

from core.target import Obstacle, Target, TargetType


# ---------------------------------------------------------------------------
# Monkey-patch: newer matplotlib versions store ``_sizes3d`` as a *list* of
# numpy scalars after projection, which breaks boolean indexing on the next
# draw. Coerce to a numpy float array before the original method runs.
# ---------------------------------------------------------------------------
_orig_do_3d_projection = Path3DCollection.do_3d_projection


def _patched_do_3d_projection(self, *args, **kwargs):
    # Coerce existing _sizes3d to ndarray (in case the user or a previous
    # call stored it as a list of numpy scalars).
    sizes = getattr(self, "_sizes3d", None)
    if sizes is not None and not isinstance(sizes, np.ndarray):
        try:
            self._sizes3d = np.asarray(sizes, dtype=float)
        except Exception:
            pass
    result = _orig_do_3d_projection(self, *args, **kwargs)
    # The original may have written a list back; coerce again defensively.
    sizes = getattr(self, "_sizes3d", None)
    if sizes is not None and not isinstance(sizes, np.ndarray):
        try:
            self._sizes3d = np.asarray(sizes, dtype=float)
        except Exception:
            pass
    return result


Path3DCollection.do_3d_projection = _patched_do_3d_projection


EventKind = str  # "target_reached" | "collision" | "pickup" | "delivery"


@dataclass
class RenderEvent:
    """A short-lived visual event that flashes on the canvas."""

    kind: EventKind
    position: np.ndarray
    drone_id: int = -1
    target_id: int = -1
    timestamp: float = 0.0
    duration: float = 0.6
    color: str = "white"


@dataclass
class RenderState:
    """Single-frame state passed to the renderer.

    Attributes
    ----------
    states : (N, 12) ndarray
        Per-drone state vectors ``[pos, vel, euler, omega]``.
    targets : list[Target]
    obstacles : list[Obstacle]
    assignment : (N,) ndarray of int
        Currently assigned target index per drone (-1 = unassigned).
    carry_status : (N,) ndarray of bool
    info : dict
        Free-form debug payload (steps, rewards, completed, etc.).
    events : list[RenderEvent]
        Optional events triggered this step (target reached, collision...).
    trail_length : int
        Maximum trail length per drone (capped to limit memory).
    """

    states: np.ndarray
    targets: List[Target]
    obstacles: List[Obstacle]
    assignment: np.ndarray
    carry_status: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)
    events: List[RenderEvent] = field(default_factory=list)
    trail_length: int = 200


@dataclass
class _ArtistBundle:
    """All reusable artists in a single renderer's scene."""

    drone_scatter: Any
    drone_labels: List[Any]
    drone_bodies: List[Any]            # quadrotor X-frame line collections
    drone_attitude: List[Any]          # forward-up arrow lines
    drone_carry_markers: List[Any]     # small package glyph above carrying drones
    trail_lines: List[Any]             # one Line3D per drone
    target_scatter: Any
    target_labels: List[Any]
    target_halo: Any                   # pulsing halo for newly reached targets
    obstacle_collections: List[Any]    # one collection per obstacle
    hud_step: Any
    hud_reward: Any
    hud_completed: Any
    hud_carry: Any
    hud_mode: Any
    title: Any
    topdown_scatter_drones: Any
    topdown_scatter_targets: Any
    topdown_scatter_obstacles: Any
    topdown_trails: List[Any]
    topdown_assignment_lines: List[Any]


class Renderer(ABC):
    """Abstract base class following the Gymnasium 1.0 render protocol."""

    metadata = {"render_fps": 30}

    def __init__(self, bounds: np.ndarray):
        self.bounds = np.asarray(bounds, dtype=float)
        self._initialized = False

    @abstractmethod
    def update(self, state: RenderState) -> None:
        """Refresh artists with new data. Must be fast (artist reuse)."""

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Return the current canvas as an RGB ndarray, or ``None``."""

    @abstractmethod
    def close(self) -> None:
        """Release all rendering resources."""

    def pause(self, dt: float) -> None:
        """Block for ``dt`` seconds to maintain a target FPS (default impl)."""
        time.sleep(max(0.0, dt))


class Matplotlib3DRenderer(Renderer):
    """Full-featured 3D renderer with artist pooling, trails and HUD.

    Parameters
    ----------
    bounds : (3, 2) ndarray
        World bounds ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``.
    show_trail : bool
        Whether to draw drone path trails.
    show_attitude : bool
        Whether to draw quadrotor body frames (uses euler angles).
    show_hud : bool
        Whether to draw the heads-up display text overlay.
    show_top_down : bool
        Whether to draw a 2D top-down inset.
    trail_length : int
        Maximum number of trail points retained per drone.
    figsize : tuple
        Figure size in inches.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        show_trail: bool = True,
        show_attitude: bool = True,
        show_hud: bool = True,
        show_top_down: bool = True,
        trail_length: int = 200,
        figsize: Tuple[float, float] = (11.0, 8.0),
    ):
        super().__init__(bounds)
        self.show_trail = show_trail
        self.show_attitude = show_attitude
        self.show_hud = show_hud
        self.show_top_down = show_top_down
        self.trail_length = trail_length
        self.figsize = figsize

        self._fig: Optional[Figure] = None
        self._ax: Optional[Axes3D] = None
        self._ax_top: Optional[Any] = None
        self._artists: Optional[_ArtistBundle] = None
        self._trails: List[Deque[np.ndarray]] = []
        self._n_drones: int = 0
        self._n_targets: int = 0
        self._event_queue: List[Tuple[RenderEvent, float]] = []
        self._mode_label: str = ""
        self._frame_size: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_canvas(self, n_drones: int, n_targets: int, n_obstacles: int):
        plt.ion()
        self._fig = plt.figure(figsize=self.figsize)
        self._ax = self._fig.add_subplot(111, projection="3d")
        x, y, z = self.bounds
        self._ax.set_xlim(x[0], x[1])
        self._ax.set_ylim(y[0], y[1])
        self._ax.set_zlim(z[0], z[1])
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._ax.set_zlabel("Z")

        if self.show_top_down:
            self._ax_top = self._fig.add_axes([0.72, 0.72, 0.25, 0.25])
            self._ax_top.set_xlim(x[0], x[1])
            self._ax_top.set_ylim(y[0], y[1])
            self._ax_top.set_aspect("equal", adjustable="box")
            self._ax_top.set_xticks([])
            self._ax_top.set_yticks([])
            self._ax_top.set_title("Top-Down", fontsize=8)

        self._init_artists(n_drones, n_targets, n_obstacles)
        self._n_drones = n_drones
        self._n_targets = n_targets

        # Force first draw so buffer_rgba has the right size.
        self._fig.canvas.draw()
        w, h = self._fig.canvas.get_width_height()
        self._frame_size = (h, w)

    def _init_artists(self, n_drones: int, n_targets: int, n_obstacles: int):
        assert self._ax is not None
        drone_colors = plt.cm.rainbow(np.linspace(0, 1, max(n_drones, 1)))

        # Drones (3D main view)
        # Initialize with the actual number of drones at a placeholder
        # position so that ``_sizes3d`` matches ``_offsets3d`` after every
        # update (avoids newer-matplotlib indexing errors).
        drone_scatter = self._ax.scatter(
            np.zeros(n_drones), np.zeros(n_drones), np.zeros(n_drones),
            c=drone_colors[:n_drones], s=80, marker="^",
            edgecolors="black", depthshade=True,
        )
        drone_labels: List[Any] = []
        drone_bodies: List[Any] = []
        drone_attitude: List[Any] = []
        drone_carry_markers: List[Any] = []
        for i in range(n_drones):
            lbl, = self._ax.plot([], [], [], "k", lw=0.5)
            lbl.set_visible(False)
            t = self._ax.text(
                0, 0, 0, f"D{i}",
                fontsize=9, color="black", fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white", edgecolor="black",
                    alpha=0.85, linewidth=0.6,
                ),
                zorder=10,
            )
            drone_labels.append(t)
            if self.show_attitude:
                body, = self._ax.plot([], [], [], "-", color=drone_colors[i],
                                      lw=1.6, alpha=0.9)
                body.set_visible(False)
                drone_bodies.append(body)
                fwd, = self._ax.plot([], [], [], "-", color=drone_colors[i],
                                     lw=1.2, alpha=0.8)
                fwd.set_visible(False)
                drone_attitude.append(fwd)
            else:
                drone_bodies.append(None)
                drone_attitude.append(None)
            carry_marker = self._ax.scatter(
                [], [], [], c="gold", s=60, marker="s", edgecolors="black",
            )
            carry_marker.set_visible(False)
            drone_carry_markers.append(carry_marker)

        # Trails
        trail_lines: List[Any] = []
        if self.show_trail:
            for i in range(n_drones):
                line, = self._ax.plot(
                    [], [], [], "-", color=drone_colors[i], lw=1.0, alpha=0.55,
                )
                trail_lines.append(line)
        else:
            trail_lines = [None] * n_drones

        # Targets
        # Initialize with the actual number of targets at a placeholder
        # position; per-target sizes vary so we set ``sizes`` here too.
        target_scatter = self._ax.scatter(
            np.zeros(n_targets), np.zeros(n_targets), np.zeros(n_targets),
            c=["tab:blue"] * n_targets, s=140, marker="o",
            edgecolors="black", alpha=0.85, depthshade=False,
        )
        target_labels: List[Any] = []
        for j in range(n_targets):
            t = self._ax.text(
                0, 0, 0, f"T{j}",
                fontsize=9, color="black", fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white", edgecolor="black",
                    alpha=0.9, linewidth=0.7,
                ),
                zorder=10,
            )
            t.set_visible(False)
            target_labels.append(t)
        target_halo = self._ax.scatter(
            [0.0], [0.0], [0.0], c=["white"], s=[400], marker="o", alpha=0.25,
            depthshade=False,
        )
        target_halo.set_visible(False)

        # Obstacles
        obstacle_collections: List[Any] = []
        for _ in range(n_obstacles):
            coll = self._obstacle_placeholder()
            obstacle_collections.append(coll)

        # HUD text
        if self.show_hud:
            hud_step = self._ax.text2D(0.02, 0.96, "", transform=self._ax.transAxes,
                                        fontsize=10, family="monospace",
                                        verticalalignment="top")
            hud_reward = self._ax.text2D(0.02, 0.90, "", transform=self._ax.transAxes,
                                          fontsize=10, family="monospace",
                                          verticalalignment="top")
            hud_completed = self._ax.text2D(0.02, 0.84, "", transform=self._ax.transAxes,
                                             fontsize=10, family="monospace",
                                             verticalalignment="top")
            hud_carry = self._ax.text2D(0.02, 0.78, "", transform=self._ax.transAxes,
                                         fontsize=10, family="monospace",
                                         verticalalignment="top")
            hud_mode = self._ax.text2D(0.98, 0.96, "", transform=self._ax.transAxes,
                                        fontsize=10, family="monospace",
                                        verticalalignment="top",
                                        horizontalalignment="right")
        else:
            hud_step = hud_reward = hud_completed = hud_carry = hud_mode = None

        title = self._ax.set_title("")

        # Top-down 2D inset artists
        if self.show_top_down:
            # Use a single dummy point to avoid newer-matplotlib errors when
            # constructing an empty scatter with color arrays of size > 0.
            td_drones = self._ax_top.scatter(
                [0.0], [0.0], c=[drone_colors[0]], s=60, marker="^",
                edgecolors="black",
            )
            td_drones.set_visible(False)
            td_targets = self._ax_top.scatter(
                [0.0], [0.0], c=["tab:blue"], s=80, marker="o",
                edgecolors="black",
            )
            td_targets.set_visible(False)
            td_obstacles = self._ax_top.scatter(
                [0.0], [0.0], c="gray", s=40, marker="s", alpha=0.5,
            )
            td_obstacles.set_visible(False)
            td_trails: List[Any] = []
            for i in range(n_drones):
                line, = self._ax_top.plot([], [], "-", color=drone_colors[i],
                                           lw=0.8, alpha=0.5)
                td_trails.append(line)
            td_assignments: List[Any] = []
            for i in range(n_drones):
                line, = self._ax_top.plot([], [], "--", color=drone_colors[i],
                                           lw=0.6, alpha=0.3)
                td_assignments.append(line)
        else:
            td_drones = td_targets = td_obstacles = None
            td_trails = []
            td_assignments = []

        self._artists = _ArtistBundle(
            drone_scatter=drone_scatter,
            drone_labels=drone_labels,
            drone_bodies=drone_bodies,
            drone_attitude=drone_attitude,
            drone_carry_markers=drone_carry_markers,
            trail_lines=trail_lines,
            target_scatter=target_scatter,
            target_labels=target_labels,
            target_halo=target_halo,
            obstacle_collections=obstacle_collections,
            hud_step=hud_step,
            hud_reward=hud_reward,
            hud_completed=hud_completed,
            hud_carry=hud_carry,
            hud_mode=hud_mode,
            title=title,
            topdown_scatter_drones=td_drones,
            topdown_scatter_targets=td_targets,
            topdown_scatter_obstacles=td_obstacles,
            topdown_trails=td_trails,
            topdown_assignment_lines=td_assignments,
        )

    def _obstacle_placeholder(self) -> Any:
        """Create a placeholder obstacle artist (Line3DCollection).

        Newer matplotlib versions reject empty ``Line3DCollection`` instances
        when added to a 3D axis, so we initialise with a single zero-length
        segment and immediately replace it via ``set_segments`` in
        :meth:`update`.
        """
        coll = Line3DCollection(
            np.zeros((1, 2, 3)), colors="gray", linewidths=0.8, alpha=0.5,
        )
        self._ax.add_collection3d(coll)
        return coll

    @staticmethod
    def _fix_scatter_sizes(scatter, expected: int) -> None:
        """Coerce 3D scatter ``_sizes3d`` to a proper numpy array.

        After ``do_3d_projection`` runs, newer matplotlib versions store
        ``_sizes3d`` as a list of ``np.float64`` scalars which cannot be
        indexed by a boolean array on subsequent ``draw`` calls. We normalise
        it back to an ``ndarray``.
        """
        sizes = getattr(scatter, "_sizes3d", None)
        if sizes is None:
            return
        if not isinstance(sizes, np.ndarray):
            try:
                arr = np.asarray(sizes, dtype=float)
            except Exception:
                return
            scatter._sizes3d = arr
        # Pad / trim to match expected point count to keep indexing in sync.
        if len(scatter._sizes3d) != expected:
            try:
                current = scatter._sizes3d
                if expected == 0:
                    scatter._sizes3d = np.zeros(0, dtype=float)
                elif len(current) == 1:
                    scatter._sizes3d = np.full(expected, float(current[0]),
                                                dtype=float)
                else:
                    # Last resort: truncate or pad with the last value.
                    if expected < len(current):
                        scatter._sizes3d = np.asarray(
                            current[:expected], dtype=float,
                        )
                    else:
                        pad = np.full(expected - len(current),
                                       float(current[-1]), dtype=float)
                        scatter._sizes3d = np.concatenate(
                            [np.asarray(current, dtype=float), pad],
                        )
            except Exception:
                pass

    def update(self, state: RenderState) -> None:
        if not self._initialized:
            self._init_canvas(
                n_drones=state.states.shape[0],
                n_targets=len(state.targets),
                n_obstacles=len(state.obstacles),
            )
            self._trails = [
                deque(maxlen=self.trail_length) for _ in range(state.states.shape[0])
            ]
            self._initialized = True
            self._mode_label = state.info.get("mode", "reach")

        # Newer matplotlib versions store 3D scatter ``_sizes3d`` as a list of
        # numpy scalars after projection, which breaks boolean indexing at the
        # next ``draw``. Convert back to a numpy float array defensively.
        self._fix_scatter_sizes(self._artists.drone_scatter,
                                 expected=state.states.shape[0])
        self._fix_scatter_sizes(self._artists.target_scatter,
                                 expected=len(state.targets))
        self._fix_scatter_sizes(self._artists.target_halo, expected=1)

        assert self._artists is not None
        artists = self._artists
        n_drones = state.states.shape[0]

        # Resize trail buffers if number of drones changed
        if len(self._trails) != n_drones:
            self._trails = [
                deque(maxlen=self.trail_length) for _ in range(n_drones)
            ]

        # Append current positions to trails
        for i in range(n_drones):
            self._trails[i].append(state.states[i, :3].copy())

        # ----- Drones -----
        positions = state.states[:, :3]
        drone_colors = self._drone_colors(n_drones)
        artists.drone_scatter._offsets3d = (
            positions[:, 0], positions[:, 1], positions[:, 2],
        )
        artists.drone_scatter.set_color(drone_colors)
        artists.drone_scatter.set_edgecolor("black")
        artists.drone_scatter.set_visible(True)

        for i in range(n_drones):
            pos = positions[i]
            # ``set_position_3d`` is required (not ``set_position``) because
            # matplotlib 3.10+ stores 3D text z-coordinate separately via
            # Text3D.set_z / Text3D.set_position_3d.
            artists.drone_labels[i].set_position_3d(
                (pos[0], pos[1], pos[2] + 0.9),
            )
            artists.drone_labels[i].set_text(
                f"D{i}{'·C' if state.carry_status[i] else ''}"
            )
            artists.drone_labels[i].set_visible(True)

            if self.show_attitude and artists.drone_bodies[i] is not None:
                xs, ys, zs = self._quadrotor_frame(state.states[i])
                artists.drone_bodies[i].set_data(xs, ys)
                artists.drone_bodies[i].set_3d_properties(zs)
                artists.drone_bodies[i].set_color(drone_colors[i])
                artists.drone_bodies[i].set_visible(True)

                fx, fy, fz = self._forward_arrow(state.states[i])
                artists.drone_attitude[i].set_data([pos[0], fx], [pos[1], fy])
                artists.drone_attitude[i].set_3d_properties([pos[2], fz])
                artists.drone_attitude[i].set_color(drone_colors[i])
                artists.drone_attitude[i].set_visible(True)

            if state.carry_status[i]:
                cm = artists.drone_carry_markers[i]
                cm._offsets3d = ([pos[0]], [pos[1]], [pos[2] + 1.0])
                cm.set_visible(True)
            else:
                artists.drone_carry_markers[i].set_visible(False)

            if self.show_trail and artists.trail_lines[i] is not None:
                if len(self._trails[i]) >= 2:
                    tra = np.array(self._trails[i])
                    artists.trail_lines[i].set_data(tra[:, 0], tra[:, 1])
                    artists.trail_lines[i].set_3d_properties(tra[:, 2])
                    artists.trail_lines[i].set_color(drone_colors[i])
                    artists.trail_lines[i].set_visible(True)
                else:
                    artists.trail_lines[i].set_visible(False)

        # ----- Targets -----
        if len(state.targets) > 0:
            t_pos = np.array([t.position for t in state.targets])
            t_colors = [self._target_color(t) for t in state.targets]
            artists.target_scatter._offsets3d = (t_pos[:, 0], t_pos[:, 1], t_pos[:, 2])
            artists.target_scatter.set_color(t_colors)
            artists.target_scatter.set_alpha(0.85)
            artists.target_scatter.set_visible(True)
            for j, t in enumerate(state.targets):
                # ``set_position_3d`` is required for correct 3D placement
                # (plain ``set_position`` discards z).
                artists.target_labels[j].set_position_3d(
                    (t.position[0], t.position[1], t.position[2] + 2.0),
                )
                artists.target_labels[j].set_text(
                    f"T{j}{'✓' if t.reached else ''}"
                )
                artists.target_labels[j].set_color("gray" if t.reached else "black")
                artists.target_labels[j].set_visible(True)
        else:
            artists.target_scatter.set_visible(False)

        # ----- Halos (event pulses) -----
        self._enqueue_events(state)
        halos = self._active_halo_points()
        if halos is not None and len(halos) > 0:
            artists.target_halo._offsets3d = (halos[:, 0], halos[:, 1], halos[:, 2])
            artists.target_halo.set_color([h[3] for h in halos])
            artists.target_halo.set_sizes([h[2] for h in halos])
            artists.target_halo.set_alpha(0.25)
            artists.target_halo.set_visible(True)
        else:
            artists.target_halo.set_visible(False)

        # ----- Obstacles -----
        for i, obs in enumerate(state.obstacles):
            if i >= len(artists.obstacle_collections):
                break
            segs = self._obstacle_segments(obs)
            artists.obstacle_collections[i].set_segments(segs)
            artists.obstacle_collections[i].set_visible(len(segs) > 0)

        # ----- HUD -----
        if self.show_hud:
            info = state.info
            artists.hud_step.set_text(
                f"step:    {info.get('steps', '?')}"
            )
            r = info.get("reward")
            if r is not None:
                try:
                    artists.hud_reward.set_text(
                        f"reward:  {float(np.sum(r)):.2f}"
                    )
                except (TypeError, ValueError):
                    artists.hud_reward.set_text("reward:  -")
            comp = info.get("completed")
            total = info.get("total_targets", len(state.targets))
            if comp is not None:
                artists.hud_completed.set_text(
                    f"targets: {int(comp)}/{int(total)}"
                )
            carry = int(np.sum(state.carry_status))
            artists.hud_carry.set_text(f"carrying:{carry}/{n_drones}")
            artists.hud_mode.set_text(
                f"mode: {self._mode_label}\nfps: {self.metadata['render_fps']}"
            )

        artists.title.set_text(
            f"Multi-UAV Delivery ({self._mode_label})"
        )

        # ----- Top-down 2D inset -----
        if self.show_top_down and artists.topdown_scatter_drones is not None:
            td = artists
            td.topdown_scatter_drones.set_offsets(positions[:, :2])
            td.topdown_scatter_drones.set_facecolor(drone_colors)
            td.topdown_scatter_drones.set_visible(True)
            if len(state.targets) > 0:
                td.topdown_scatter_targets.set_offsets(t_pos[:, :2])
                td.topdown_scatter_targets.set_facecolor(t_colors)
                td.topdown_scatter_targets.set_visible(True)
            else:
                td.topdown_scatter_targets.set_visible(False)
            if state.obstacles:
                obs_xy = np.array([o.position[:2] for o in state.obstacles])
                td.topdown_scatter_obstacles.set_offsets(obs_xy)
                td.topdown_scatter_obstacles.set_visible(True)
            else:
                td.topdown_scatter_obstacles.set_visible(False)
            for i in range(n_drones):
                if i < len(td.topdown_trails) and len(self._trails[i]) >= 2:
                    tra = np.array(self._trails[i])
                    td.topdown_trails[i].set_data(tra[:, 0], tra[:, 1])
                    td.topdown_trails[i].set_visible(True)
                elif i < len(td.topdown_trails):
                    td.topdown_trails[i].set_visible(False)
                if i < len(td.topdown_assignment_lines):
                    aid = state.assignment[i]
                    if 0 <= aid < len(state.targets):
                        target_pos = state.targets[aid].position[:2]
                        td.topdown_assignment_lines[i].set_data(
                            [positions[i, 0], target_pos[0]],
                            [positions[i, 1], target_pos[1]],
                        )
                        td.topdown_assignment_lines[i].set_color(drone_colors[i])
                        td.topdown_assignment_lines[i].set_visible(True)
                    else:
                        td.topdown_assignment_lines[i].set_data([], [])
                        td.topdown_assignment_lines[i].set_visible(False)

        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Frame export
    # ------------------------------------------------------------------

    def get_frame(self) -> Optional[np.ndarray]:
        if self._fig is None:
            return None
        self._fig.canvas.draw()
        try:
            buf = np.asarray(self._fig.canvas.buffer_rgba())
        except Exception:
            return None
        if buf.ndim != 3 or buf.shape[2] != 4:
            return None
        return buf[:, :, :3].copy()

    def close(self) -> None:
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._ax_top = None
            self._artists = None
            self._trails = []
            self._initialized = False

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _drone_colors(self, n: int) -> List[Any]:
        return list(plt.cm.rainbow(np.linspace(0, 1, max(n, 1))))

    @staticmethod
    def _target_color(t: Target) -> str:
        if t.reached:
            return "gray"
        return {
            TargetType.REACH: "tab:blue",
            TargetType.PICKUP: "tab:green",
            TargetType.DELIVERY: "tab:red",
        }.get(t.target_type, "tab:blue")

    def _quadrotor_frame(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the 4-rotor X-frame of a quadrotor in world coords.

        The body frame is rotated by the drone's Euler angles (ZYX).
        """
        pos = state[0:3]
        euler = state[6:9]
        arm = 0.4  # visual arm length, not physical
        # Body-frame rotor offsets (X-config)
        offsets_body = np.array([
            [ arm, 0.0, 0.0],
            [-arm, 0.0, 0.0],
            [ 0.0,  arm, 0.0],
            [ 0.0, -arm, 0.0],
        ])
        R = self._euler_to_R(euler)
        offsets_world = offsets_body @ R.T
        pts = offsets_world + pos
        # Connect 0-2-1-3-0 (X-frame)
        order = [0, 2, 1, 3, 0]
        return pts[order, 0], pts[order, 1], pts[order, 2]

    def _forward_arrow(self, state: np.ndarray, length: float = 0.8) -> Tuple[float, float, float]:
        pos = state[0:3]
        euler = state[6:9]
        R = self._euler_to_R(euler)
        forward_body = np.array([length, 0.0, 0.0])
        forward_world = R @ forward_body
        return pos[0] + forward_world[0], pos[1] + forward_world[1], pos[2] + forward_world[2]

    @staticmethod
    def _euler_to_R(euler: np.ndarray) -> np.ndarray:
        """ZYX intrinsic Euler -> rotation matrix (body -> world)."""
        phi, theta, psi = euler
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        return np.array([
            [c_psi * c_theta, c_psi * s_theta * s_phi - s_psi * c_phi,
             c_psi * s_theta * c_phi + s_psi * s_phi],
            [s_psi * c_theta, s_psi * s_theta * s_phi + c_psi * c_phi,
             s_psi * s_theta * c_phi - c_psi * s_phi],
            [-s_theta, c_theta * s_phi, c_theta * c_phi],
        ])

    @staticmethod
    def _obstacle_segments(obs: Obstacle):
        """Return line segments (Nx2x3) for an obstacle, in world coords."""
        c = obs.position
        if obs.shape == "cylinder":
            r = obs.size[0]
            h = obs.size[2]
            theta = np.linspace(0, 2 * np.pi, 24)
            x = c[0] + r * np.cos(theta)
            y = c[1] + r * np.sin(theta)
            z_bot = c[2] - h / 2
            z_top = c[2] + h / 2
            segs = []
            # top and bottom rings
            for k in range(len(theta) - 1):
                segs.append([[x[k], y[k], z_bot], [x[k + 1], y[k + 1], z_bot]])
                segs.append([[x[k], y[k], z_top], [x[k + 1], y[k + 1], z_top]])
            # a few vertical struts
            for k in range(0, len(theta), 4):
                segs.append([[x[k], y[k], z_bot], [x[k], y[k], z_top]])
            return segs
        if obs.shape == "sphere":
            r = obs.size[0]
            u = np.linspace(0, 2 * np.pi, 12)
            v = np.linspace(0, np.pi, 8)
            segs = []
            for v0 in v:
                xs = c[0] + r * np.cos(u) * np.sin(v0)
                ys = c[1] + r * np.sin(u) * np.sin(v0)
                zs = np.full_like(xs, c[2] + r * np.cos(v0))
                for k in range(len(u) - 1):
                    segs.append([[xs[k], ys[k], zs[k]], [xs[k + 1], ys[k + 1], zs[k + 1]]])
            for u0 in u:
                xs = c[0] + r * np.cos(u0) * np.sin(v)
                ys = c[1] + r * np.sin(u0) * np.sin(v)
                zs = c[2] + r * np.cos(v)
                for k in range(len(v) - 1):
                    segs.append([[xs[k], ys[k], zs[k]], [xs[k + 1], ys[k + 1], zs[k + 1]]])
            return segs
        if obs.shape == "box":
            sx, sy, sz = obs.size
            x = c[0] + np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * sx / 2
            y = c[1] + np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * sy / 2
            z = c[2] + np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * sz / 2
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            return [[[x[a], y[a], z[a]], [x[b], y[b], z[b]]] for a, b in edges]
        return []

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _enqueue_events(self, state: RenderState) -> None:
        now = time.monotonic()
        for ev in state.events:
            ev.timestamp = now
            self._event_queue.append((ev, now))
        # GC expired events
        self._event_queue = [
            (ev, t) for ev, t in self._event_queue if (now - t) < ev.duration
        ]

    def _active_halo_points(self) -> Optional[np.ndarray]:
        if not self._event_queue:
            return None
        now = time.monotonic()
        rows = []
        for ev, t in self._event_queue:
            age = now - t
            if age >= ev.duration:
                continue
            ratio = 1.0 - age / ev.duration
            size = 200 + 600 * ratio
            color = ev.color
            rows.append([ev.position[0], ev.position[1], ev.position[2], color, size])
        if not rows:
            return None
        arr = np.array(rows, dtype=object)
        return arr


# ----------------------------------------------------------------------
# Backward-compatible legacy API
# ----------------------------------------------------------------------


class SimpleRenderer(Matplotlib3DRenderer):
    """Deprecated thin wrapper over :class:`Matplotlib3DRenderer`.

    The original implementation redrew the entire scene on every call. The new
    implementation pools matplotlib artists and only updates their data, which
    is several times faster. This class preserves the old call signature so
    external scripts keep working.
    """

    def __init__(self, bounds: np.ndarray, *args, **kwargs):
        warnings.warn(
            "SimpleRenderer is deprecated; use Matplotlib3DRenderer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(bounds, *args, **kwargs)

    def render(
        self,
        states: np.ndarray,
        targets: List[Target],
        obstacles: List[Obstacle],
        assignment: np.ndarray,
        mode: str = "human",
    ) -> Optional[np.ndarray]:
        """Legacy single-call renderer.

        ``mode`` follows the original semantics:
            * ``"human"``     -> draw to the interactive window, return ``None``.
            * ``"rgb_array"`` -> return the current frame as ``(H, W, 3)`` uint8.
        """
        state = RenderState(
            states=states,
            targets=targets,
            obstacles=obstacles,
            assignment=assignment,
            carry_status=np.zeros(states.shape[0], dtype=bool),
        )
        self.update(state)
        if mode == "rgb_array":
            return self.get_frame()
        return None


def visualize_sample_scene():
    """Standalone scene visualization for testing (backward compatible)."""
    from core.target import generate_random_targets, generate_random_obstacles
    from core.dynamics import QuadrotorDynamics

    bounds = np.array([[-30, 30], [-30, 30], [0, 30]])
    rng = np.random.default_rng(42)

    obstacles = generate_random_obstacles(8, bounds, rng=rng)
    drone_positions = np.array([
        [-20, -20, 1], [-20, 20, 1], [20, -20, 1], [20, 20, 1],
    ])
    targets = generate_random_targets(
        4, bounds, mode="delivery",
        drone_positions=drone_positions,
        obstacle_positions=np.array([o.position for o in obstacles]),
        obstacle_radii=np.array([o.radius for o in obstacles]),
        rng=rng,
    )

    dynamics = QuadrotorDynamics()
    states = np.array([dynamics.reset_state(drone_positions[i]) for i in range(4)])
    assignment = np.array([0, 1, 2, 3])

    renderer = SimpleRenderer(bounds)
    renderer.render(states, targets, obstacles, assignment, mode="human")
    time.sleep(3)
    renderer.close()
    print("Visualization test complete.")


if __name__ == "__main__":
    visualize_sample_scene()
