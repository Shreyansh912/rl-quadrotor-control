import numpy as np


class QuadrotorDynamics:
    """
    6-DOF Nonlinear Rigid-Body Quadrotor Dynamics using Newton-Euler equations
    with Quaternion orientation representation.
    """
    def __init__(
        self,
        mass: float = 0.85,          # kg
        arm_length: float = 0.22,     # meters (center to motor)
        Ixx: float = 4.85e-3,        # kg*m^2
        Iyy: float = 4.85e-3,        # kg*m^2
        Izz: float = 8.80e-3,        # kg*m^2
        kf: float = 1.2e-5,          # Thrust coefficient (N / (rad/s)^2)
        km: float = 2.4e-7,          # Drag torque coefficient (N*m / (rad/s)^2)
        motor_tau: float = 0.03,     # Motor first-order lag time constant (seconds)
        max_rpm: float = 12000.0,    # Max motor RPM
    ):
        self.mass = mass
        self.arm_length = arm_length
        self.g = 9.81

        # Inertia matrix
        self.I = np.diag([Ixx, Iyy, Izz])
        self.inv_I = np.linalg.inv(self.I)

        self.kf = kf
        self.km = km
        self.motor_tau = motor_tau
        self.max_omega = (max_rpm * 2.0 * np.pi) / 60.0  # rad/s

        # Arm offset projection (X-configuration)
        self.d = arm_length / np.sqrt(2.0)

        # Allocation matrix mapping motor thrusts [T1, T2, T3, T4] to [Total Thrust, Tau_x, Tau_y, Tau_z]
        self.allocation_matrix = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-self.d, self.d, self.d, -self.d],
            [self.d, -self.d, self.d, -self.d],
            [-km / kf, -km / kf, km / kf, km / kf]
        ], dtype=np.float64)

        self.reset()

    def reset(self, init_pos=np.zeros(3), init_vel=np.zeros(3)):
        self.pos = np.array(init_pos, dtype=np.float64)
        self.vel = np.array(init_vel, dtype=np.float64)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)
        self.motor_speeds = np.zeros(4, dtype=np.float64)

    def quat_to_rot_matrix(self, q):
        """Converts unit quaternion to 3x3 SO(3) rotation matrix."""
        qw, qx, qy, qz = q
        return np.array([
            [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx**2 + qy**2)]
        ], dtype=np.float64)

    def quat_derivative(self, q, omega):
        """Computes time derivative of quaternion given body angular rates."""
        qw, qx, qy, qz = q
        p, q_rate, r = omega
        q_dot = 0.5 * np.array([
            -qx * p - qy * q_rate - qz * r,
             qw * p + qy * r - qz * q_rate,
             qw * q_rate - qx * r + qz * p,
             qw * r + qx * q_rate - qy * p
        ], dtype=np.float64)
        return q_dot

    def step(self, motor_commands, dt=0.01, external_force=np.zeros(3), external_torque=np.zeros(3)):
        """
        Integrates non-linear equations of motion by dt.
        motor_commands: 4 values in range [0, 1] representing normalized throttle setpoints.
        """
        target_motor_speeds = np.clip(motor_commands, 0.0, 1.0) * self.max_omega
        self.motor_speeds += (target_motor_speeds - self.motor_speeds) * (dt / max(self.motor_tau, 1e-4))

        motor_thrusts = self.kf * (self.motor_speeds**2)
        wrench = self.allocation_matrix @ motor_thrusts
        f_total = wrench[0]
        torques = wrench[1:4] + external_torque

        R = self.quat_to_rot_matrix(self.quat)
        thrust_world = R @ np.array([0.0, 0.0, f_total])
        gravity_world = np.array([0.0, 0.0, -self.mass * self.g])
        drag_force = -0.15 * np.linalg.norm(self.vel) * self.vel

        acc = (thrust_world + gravity_world + external_force + drag_force) / self.mass

        cross_term = np.cross(self.omega, self.I @ self.omega)
        omega_dot = self.inv_I @ (torques - cross_term)

        self.pos += self.vel * dt + 0.5 * acc * (dt**2)
        self.vel += acc * dt

        q_dot = self.quat_derivative(self.quat, self.omega)
        self.quat += q_dot * dt
        self.quat /= np.linalg.norm(self.quat)

        self.omega += omega_dot * dt

        return {
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "quat": self.quat.copy(),
            "omega": self.omega.copy(),
            "thrust": f_total
        }