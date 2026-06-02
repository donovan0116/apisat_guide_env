"""
Simple Matplotlib-based 3D renderer for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Optional
import time

from core.target import Target, TargetType, Obstacle


class SimpleRenderer:
    """Matplotlib-based 3D visualization."""

    def __init__(self, bounds: np.ndarray):
        self.bounds = bounds
        self._fig = None
        self._ax = None
        self._initialized = False

    def render(
        self,
        states: np.ndarray,              # [n_drones, 12]
        targets: List[Target],
        obstacles: List[Obstacle],
        assignment: np.ndarray,          # [n_drones]
        mode: str = "human",
    ) -> Optional[np.ndarray]:
        if not self._initialized:
            self._init_plot()
            self._initialized = True

        self._ax.clear()
        self._draw_bounds()
        self._draw_obstacles(obstacles)
        self._draw_targets(targets)
        self._draw_drones(states, assignment, targets)
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._ax.set_zlabel("Z")
        self._ax.set_title("Multi-UAV Delivery Task Assignment")

        self._fig.canvas.draw()

        if mode == "human":
            plt.pause(0.01)
            return None
        elif mode == "rgb_array":
            self._fig.canvas.draw()
            img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return img

        return None

    def _init_plot(self):
        plt.ion()
        self._fig = plt.figure(figsize=(10, 8))
        self._ax = self._fig.add_subplot(111, projection="3d")

    def _draw_bounds(self):
        x = self.bounds[0]
        y = self.bounds[1]
        z = self.bounds[2]
        self._ax.set_xlim(x[0], x[1])
        self._ax.set_ylim(y[0], y[1])
        self._ax.set_zlim(z[0], z[1])

    def _draw_obstacles(self, obstacles: List[Obstacle]):
        for obs in obstacles:
            if obs.shape == "cylinder":
                self._draw_cylinder(obs.position, obs.size[0], obs.size[2])
            elif obs.shape == "sphere":
                self._draw_sphere(obs.position, obs.size[0])
            elif obs.shape == "box":
                self._draw_box(obs.position, obs.size)

    def _draw_cylinder(self, center: np.ndarray, radius: float, height: float):
        z_bottom = center[2] - height / 2
        z_top = center[2] + height / 2
        theta = np.linspace(0, 2 * np.pi, 30)
        x_circle = center[0] + radius * np.cos(theta)
        y_circle = center[1] + radius * np.sin(theta)

        # Draw top and bottom circles
        self._ax.plot(x_circle, y_circle, z_bottom, color="gray", alpha=0.5)
        self._ax.plot(x_circle, y_circle, z_top, color="gray", alpha=0.5)

        # Draw vertical lines
        for i in range(0, 30, 6):
            self._ax.plot(
                [x_circle[i], x_circle[i]],
                [y_circle[i], y_circle[i]],
                [z_bottom, z_top],
                color="gray", alpha=0.3,
            )

    def _draw_sphere(self, center: np.ndarray, radius: float):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        self._ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

    def _draw_box(self, center: np.ndarray, size: np.ndarray):
        x = center[0] + np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * size[0] / 2
        y = center[1] + np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * size[1] / 2
        z = center[2] + np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * size[2] / 2
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]
        for edge in edges:
            self._ax.plot(
                [x[edge[0]], x[edge[1]]],
                [y[edge[0]], y[edge[1]]],
                [z[edge[0]], z[edge[1]]],
                color="gray", alpha=0.5,
            )

    def _draw_targets(self, targets: List[Target]):
        for i, t in enumerate(targets):
            color = {
                TargetType.REACH: "blue",
                TargetType.PICKUP: "green",
                TargetType.DELIVERY: "red",
            }[t.target_type]
            alpha = 0.3 if t.reached else 0.8
            marker = "o" if not t.reached else "x"
            self._ax.scatter(*t.position, c=color, s=100, alpha=alpha, marker=marker)
            self._ax.text(t.position[0], t.position[1], t.position[2],
                          f"T{i}", fontsize=8)

    def _draw_drones(self, states: np.ndarray, assignment: np.ndarray,
                     targets: List[Target]):
        n = len(states)
        target_positions = np.array([t.position for t in targets])
        colors = plt.cm.rainbow(np.linspace(0, 1, n))
        for i in range(n):
            pos = states[i, :3]
            self._ax.scatter(*pos, c=[colors[i]], s=80, marker="^", edgecolors="black")
            self._ax.text(pos[0], pos[1], pos[2], f"D{i}", fontsize=8)

            # Draw assignment line
            aid = assignment[i]
            if aid >= 0 and aid < len(target_positions):
                self._ax.plot(
                    [pos[0], target_positions[aid][0]],
                    [pos[1], target_positions[aid][1]],
                    [pos[2], target_positions[aid][2]],
                    color=colors[i], alpha=0.2, linestyle="--",
                )

    def close(self):
        if self._fig:
            plt.close(self._fig)
            self._fig = None
            self._initialized = False

# Quick test function
def visualize_sample_scene():
    """Standalone scene visualization for testing."""
    from core.target import generate_random_targets, generate_random_obstacles
    from core.dynamics import QuadrotorDynamics

    bounds = np.array([[-30, 30], [-30, 30], [0, 30]])
    rng = np.random.default_rng(42)

    obstacles = generate_random_obstacles(8, bounds, rng=rng)
    drone_positions = np.array([
        [-20, -20, 1], [-20, 20, 1], [20, -20, 1], [20, 20, 1],
    ])
    targets = generate_random_targets(4, bounds, mode="delivery",
                                       drone_positions=drone_positions,
                                       obstacle_positions=np.array([o.position for o in obstacles]),
                                       obstacle_radii=np.array([o.radius for o in obstacles]),
                                       rng=rng)

    dynamics = QuadrotorDynamics()
    states = np.array([
        dynamics.reset_state(drone_positions[i]) for i in range(4)
    ])
    assignment = np.array([0, 1, 2, 3])

    renderer = SimpleRenderer(bounds)
    renderer.render(states, targets, obstacles, assignment, mode="human")
    time.sleep(3)
    renderer.close()
    print("Visualization test complete.")


if __name__ == "__main__":
    visualize_sample_scene()
