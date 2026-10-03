import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dynamics.quadrotor_dynamics import QuadrotorDynamics


class QuadrotorTrajectoryEnv(gym.Env):
    """
    Gymnasium Environment for Dynamic 3D Trajectory Tracking (Figure-8 Lemniscate).
    Observation Space (15D):
      [pos_error (3), vel_error (3), quat_rot_error (3), body_rates (3), target_pos (3)]
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, dt=0.02, max_steps=600):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0
        self.dynamics = QuadrotorDynamics()

        # Lemniscate (Figure-8) trajectory parameters
        self.radius_x = 1.0
        self.radius_y = 0.8
        self.base_z = 1.5
        self.omega_traj = 0.8  # Trajectory frequency (rad/s)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # 15-dimensional observation space (includes trajectory feedforward)
        obs_high = np.array([
            5.0, 5.0, 5.0,      # pos error (m)
            10.0, 10.0, 10.0,   # vel error (m/s)
            1.0, 1.0, 1.0,      # attitude quat error
            20.0, 20.0, 20.0,   # angular rates (rad/s)
            5.0, 5.0, 5.0       # reference target pos (m)
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

    def get_reference_trajectory(self, t):
        """Computes reference 3D position and velocity on a Lemniscate path."""
        theta = self.omega_traj * t
        
        # Position p*(t)
        x_ref = self.radius_x * np.sin(theta)
        y_ref = self.radius_y * np.sin(theta) * np.cos(theta)
        z_ref = self.base_z + 0.2 * np.sin(0.5 * theta)
        pos_ref = np.array([x_ref, y_ref, z_ref])

        # Velocity v*(t)
        vx_ref = self.radius_x * self.omega_traj * np.cos(theta)
        vy_ref = self.radius_y * self.omega_traj * (np.cos(theta)**2 - np.sin(theta)**2)
        vz_ref = 0.1 * self.omega_traj * np.cos(0.5 * theta)
        vel_ref = np.array([vx_ref, vy_ref, vz_ref])

        return pos_ref, vel_ref

    def _get_obs(self):
        t = self.step_count * self.dt
        p_ref, v_ref = self.get_reference_trajectory(t)

        pos_err = p_ref - self.dynamics.pos
        vel_err = v_ref - self.dynamics.vel
        quat_err = self.dynamics.quat[1:4]
        omega = self.dynamics.omega

        obs = np.concatenate([pos_err, vel_err, quat_err, omega, p_ref], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Start close to trajectory origin
        p_ref_0, _ = self.get_reference_trajectory(0.0)
        init_pos = p_ref_0 + np.random.uniform(-0.1, 0.1, size=3)
        init_vel = np.random.uniform(-0.05, 0.05, size=3)
        self.dynamics.reset(init_pos=init_pos, init_vel=init_vel)

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        t = self.step_count * self.dt
        p_ref, v_ref = self.get_reference_trajectory(t)

        norm_actions = np.clip((action + 1.0) / 2.0, 0.0, 1.0)
        state = self.dynamics.step(norm_actions, dt=self.dt)

        pos_err = np.linalg.norm(p_ref - state["pos"])
        vel_err = np.linalg.norm(v_ref - state["vel"])
        tilt = np.linalg.norm(state["quat"][1:3])
        ang_rate = np.linalg.norm(state["omega"])

        # Dense tracking reward
        r_pos = np.exp(-3.0 * pos_err)
        r_vel = np.exp(-1.5 * vel_err)
        r_att = np.exp(-3.0 * tilt)

        reward = (
            0.50 * r_pos
            + 0.25 * r_vel
            + 0.25 * r_att
            - 0.03 * ang_rate
            - 0.30 * (pos_err ** 2)
        )

        terminated = False
        if state["pos"][2] <= 0.25 or pos_err > 2.0 or tilt > 0.70:
            reward -= 5.0
            terminated = True

        truncated = self.step_count >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, {"pos_err": pos_err}
