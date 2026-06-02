"""
Quadrotor dynamics model based on simplified Newton-Euler equations.

State vector (13-DOF):
    [x, y, z,  vx, vy, vz,  roll, pitch, yaw,  wx, wy, wz]

Rigid-body equations:
    p_dot = v                                    (world frame)
    v_dot = (1/m) * R * F_b + g                  (world frame, F_b = [0,0,T])
    euler_dot = W_euler * omega                  (body rates -> Euler rates)
    omega_dot = J^{-1} * (tau - omega x J*omega) (body frame)

where:
    R = rotation matrix from body to world (ZYX Euler)
    g = [0, 0, -g]
    J = diag(Ixx, Iyy, Izz)
    tau = [tau_x, tau_y, tau_z]  (body-frame torques)
    T = total thrust (body-frame, along z-axis)
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class QuadrotorParams:
    """Physical parameters for a quadrotor (Crazyflie 2.X defaults)."""
    mass: float = 0.027  # kg
    arm_length: float = 0.0397  # m
    Ixx: float = 1.4e-5  # kg*m^2
    Iyy: float = 1.4e-5  # kg*m^2
    Izz: float = 2.17e-5  # kg*m^2
    kf: float = 3.16e-10  # thrust coefficient
    km: float = 7.94e-12  # torque coefficient
    max_thrust: float = 0.6  # N (total)
    g: float = 9.81  # m/s^2
    dt: float = 1.0 / 240.0  # physics time step (240 Hz)
    drag_coeff: float = 0.0  # linear drag coefficient (simplified)


class QuadrotorDynamics:
    """
    Simplified Newton-Euler quadrotor dynamics.

    Controls: [thrust, tau_x, tau_y, tau_z] in body frame.
    State: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz].

    Euler angle convention: ZYX (yaw -> pitch -> roll), intrinsic rotations.
    """

    def __init__(self, params: QuadrotorParams = None):
        self.p = params or QuadrotorParams()
        self._J = np.diag([self.p.Ixx, self.p.Iyy, self.p.Izz])
        self._J_inv = np.diag([1.0 / self.p.Ixx, 1.0 / self.p.Iyy, 1.0 / self.p.Izz])
        self._hover_thrust = self.p.mass * self.p.g  # thrust needed to hover

    @property
    def hover_thrust(self) -> float:
        return self._hover_thrust

    @property
    def state_dim(self) -> int:
        return 12

    @property
    def action_dim(self) -> int:
        return 4

    def reset_state(
        self, position: np.ndarray, velocity: np.ndarray = None,
        euler: np.ndarray = None, angular_vel: np.ndarray = None
    ) -> np.ndarray:
        """Create initial state vector."""
        vel = np.zeros(3) if velocity is None else velocity
        ang = np.zeros(3) if euler is None else euler
        omega = np.zeros(3) if angular_vel is None else angular_vel
        return np.concatenate([position, vel, ang, omega])

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Forward simulate one time step.

        Args:
            state:  [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
            action: [thrust, tau_x, tau_y, tau_z] in physical units (N, N*m)

        Returns:
            next_state: same format
        """
        pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9]
        omega = state[9:12]

        thrust = np.clip(action[0], 0.0, self.p.max_thrust)
        torque = action[1:4]

        # -- world-frame acceleration --
        R = self._rotation_matrix(euler)
        force_body = np.array([0.0, 0.0, thrust])
        force_world = R @ force_body
        acc_world = force_world / self.p.mass + np.array([0.0, 0.0, -self.p.g])

        # -- simple linear drag --
        acc_world -= self.p.drag_coeff * vel

        # -- body-frame angular acceleration --
        omega_cross = np.cross(omega, self._J @ omega)
        omega_dot = self._J_inv @ (torque - omega_cross)

        # -- Euler angle rates (ZYX convention) --
        euler_dot = self._euler_rates(euler, omega)

        # -- Euler integration --
        dt = self.p.dt
        new_pos = pos + vel * dt
        new_vel = vel + acc_world * dt
        new_euler = euler + euler_dot * dt
        new_omega = omega + omega_dot * dt

        # --- Clamping for numerical stability ---
        max_vel = self.p.pos_bound if hasattr(self.p, 'pos_bound') else 50.0
        max_omega = self.p.ang_vel_bound if hasattr(self.p, 'ang_vel_bound') else 20.0
        new_vel = np.clip(new_vel, -max_vel, max_vel)
        new_omega = np.clip(new_omega, -max_omega, max_omega)

        # wrap yaw to [-pi, pi]
        new_euler[2] = self._wrap_angle(new_euler[2])
        new_euler[0] = self._wrap_angle(new_euler[0])
        new_euler[1] = np.clip(new_euler[1], -np.pi / 2, np.pi / 2)

        return np.concatenate([new_pos, new_vel, new_euler, new_omega])

    def _rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """ZYX intrinsic rotation matrix (body -> world)."""
        phi, theta, psi = euler
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        return np.array([
            [c_psi * c_theta, c_psi * s_theta * s_phi - s_psi * c_phi, c_psi * s_theta * c_phi + s_psi * s_phi],
            [s_psi * c_theta, s_psi * s_theta * s_phi + c_psi * c_phi, s_psi * s_theta * c_phi - c_psi * s_phi],
            [-s_theta, c_theta * s_phi, c_theta * c_phi],
        ])

    def _euler_rates(self, euler: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Convert body angular velocity to Euler angle rates (ZYX)."""
        phi, theta = euler[0], euler[1]
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)

        # Transformation matrix W (ZYX convention)
        # [phi_dot, theta_dot, psi_dot]^T = W * [wx, wy, wz]^T
        if abs(c_theta) < 1e-8:
            c_theta = 1e-8 * np.sign(c_theta) if c_theta != 0 else 1e-8

        t_theta = s_theta / c_theta  # tan(theta)

        W = np.array([
            [1.0, s_phi * t_theta, c_phi * t_theta],
            [0.0, c_phi, -s_phi],
            [0.0, s_phi / c_theta, c_phi / c_theta],
        ])
        return W @ omega

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi
