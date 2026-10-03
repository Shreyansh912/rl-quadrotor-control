import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dynamics.quadrotor_dynamics import QuadrotorDynamics


class QuadrotorHoverEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, target_pos=np.array([0.0, 0.0, 1.5]), dt=0.02, max_steps=500):
        super().__init__()
        self.target_pos = np.array(target_pos, dtype=np.float64)
        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0
        self.dynamics = QuadrotorDynamics()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        obs_high = np.array([
            5.0, 5.0, 5.0,      # pos error bounds (m)
            10.0, 10.0, 10.0,   # vel bounds (m/s)
            1.0, 1.0, 1.0,      # quaternion attitude error
            20.0, 20.0, 20.0    # angular velocity bounds (rad/s)
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

    def _get_obs(self):
        pos_err = self.target_pos - self.dynamics.pos
        vel = self.dynamics.vel
        quat_err = self.dynamics.quat[1:4]
        omega = self.dynamics.omega

        obs = np.concatenate([pos_err, vel, quat_err, omega], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Uniform initial spawn around target
        init_pos = np.random.uniform(low=[-0.25, -0.25, 1.2], high=[0.25, 0.25, 1.7])
        init_vel = np.random.uniform(low=[-0.05, -0.05, -0.05], high=[0.05, 0.05, 0.05])
        self.dynamics.reset(init_pos=init_pos, init_vel=init_vel)

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        norm_actions = np.clip((action + 1.0) / 2.0, 0.0, 1.0)

        state = self.dynamics.step(norm_actions, dt=self.dt)

        # Errors
        pos_err_vec = self.target_pos - state["pos"]
        pos_err = np.linalg.norm(pos_err_vec)
        z_err = abs(pos_err_vec[2])
        vel_norm = np.linalg.norm(state["vel"])
        tilt = np.linalg.norm(state["quat"][1:3])
        ang_rate = np.linalg.norm(state["omega"])

        # Shaped exponential rewards + quadratic altitude constraint
        r_pos = np.exp(-2.5 * pos_err)
        r_z = np.exp(-4.0 * z_err)
        r_att = np.exp(-4.0 * tilt)

        # Normalized step reward in range [0, ~1.0] to prevent exploding value loss
        reward = (
            0.45 * r_pos
            + 0.35 * r_z
            + 0.20 * r_att
            - 0.05 * vel_norm
            - 0.02 * ang_rate
            - 0.50 * (z_err ** 2)
        )

        terminated = False
        if state["pos"][2] <= 0.25 or pos_err > 2.5 or tilt > 0.60:
            reward -= 5.0
            terminated = True

        truncated = self.step_count >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, {"pos_err": pos_err, "z": state["pos"][2]}