"""
Geometric utility functions.
"""

import numpy as np
from typing import Tuple


def distance_point_to_line(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> float:
    """Shortest distance from a point to a line segment (3D)."""
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq < 1e-8:
        return np.linalg.norm(point_vec)

    t = np.dot(point_vec, line_vec) / line_len_sq
    t = np.clip(t, 0.0, 1.0)

    closest = line_start + t * line_vec
    return np.linalg.norm(point - closest)


def check_line_obstacle_collision(
    start: np.ndarray,
    end: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    clearance: float = 0.5,
) -> bool:
    """Check if a line segment collides with any cylindrical obstacles."""
    for i in range(len(obstacle_positions)):
        obs_center = obstacle_positions[i, :2]
        line_start = start[:2]
        line_end = end[:2]
        d = distance_point_to_line(obs_center, line_start, line_end)
        if d < obstacle_radii[i] + clearance:
            return True
    return False


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def clip_vector(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip vector magnitude to max_norm."""
    norm = np.linalg.norm(v)
    if norm > max_norm:
        return v * max_norm / norm
    return v


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX intrinsic rotation matrix (body -> world)."""
    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    c_y, s_y = np.cos(yaw), np.sin(yaw)

    return np.array([
        [c_y * c_p, c_y * s_p * s_r - s_y * c_r, c_y * s_p * c_r + s_y * s_r],
        [s_y * c_p, s_y * s_p * s_r + c_y * c_r, s_y * s_p * c_r - c_y * s_r],
        [-s_p, c_p * s_r, c_p * c_r],
    ])
