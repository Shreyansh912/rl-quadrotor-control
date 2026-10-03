import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dynamics.quadrotor_dynamics import QuadrotorDynamics


class QuadrotorRandomizedEnv(gym.Env):
    """
    Gymnasium Environment with Full Physical Domain Randomization (Sim-to-Real).
    Randomizes mass, inertia tensor, motor lag, and asymmetric rotor degradation.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, target_pos=np.array([0.0, 0.0, 1.5]), dt=0.02, max_steps=500):
        super().__init__()
        self.target_pos = np.array(target_pos, dtype=np.float64)
        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0
        self.dynamics = QuadrotorDynamics()

        # Baseline physical parameters for restoration
        self.nominal_mass = self.dynamics.mass
        self.nominal_I = np.copy(self.dynamics.I)
        self.nominal_tau = getattr(self.dynamics, "tau", 0.03)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        obs_high = np.array([
            5.0, 5.0, 5.0,      # pos error (m)
            10.0, 10.0, 10.0,   # linear vel (m/s)
            1.0, 1.0, 1.0,      # quaternion attitude error
            20.0, 20.0, 20.0    # body angular rates (rad/s)
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Dynamic motor scale factor for asymmetry
        self.motor_scales = np.ones(4)

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

        # --- Domain Randomization Sampling ---
        # 1. Mass variation (+/- 15%)
        self.dynamics.mass = float(np.random.uniform(0.70, 0.95))

        # 2. Diagonal inertia scaling (+/- 20%)
        inertia_scale = np.random.uniform(0.80, 1.20, size=3)
        self.dynamics.I = np.diag(np.diag(self.nominal_I) * inertia_scale)
        self.dynamics.I_inv = np.linalg.inv(self.dynamics.I)

        # 3. Motor lag time constant (20ms to 50ms)
        self.dynamics.tau = float(np.random.uniform(0.02, 0.05))

        # 4. Asymmetric individual rotor thrust scaling (+/- 10%)
        self.motor_scales = np.random.uniform(0.90, 1.10, size=4)

        # Initial spawn state
        init_pos = np.random.uniform(low=[-0.25, -0.25, 1.2], high=[0.25, 0.25, 1.7])
        init_vel = np.random.uniform(low=[-0.05, -0.05, -0.05], high=[0.05, 0.05, 0.05])
        self.dynamics.reset(init_pos=init_pos, init_vel=init_vel)

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        
        # Apply rotor asymmetry scaling
        norm_actions = np.clip((action + 1.0) / 2.0, 0.0, 1.0) * self.motor_scales
        norm_actions = np.clip(norm_actions, 0.0, 1.0)

        # Step dynamics with stochastic crosswind turbulence
        wind_gust = np.random.normal(0.0, 0.4, size=3)
        state = self.dynamics.step(norm_actions, dt=self.dt, external_force=wind_gust)

        pos_err_vec = self.target_pos - state["pos"]
        pos_err = np.linalg.norm(pos_err_vec)
        z_err = abs(pos_err_vec[2])
        vel_norm = np.linalg.norm(state["vel"])
        tilt = np.linalg.norm(state["quat"][1:3])
        ang_rate = np.linalg.norm(state["omega"])

        r_pos = np.exp(-2.5 * pos_err)
        r_z = np.exp(-4.0 * z_err)
        r_att = np.exp(-4.0 * tilt)

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
        return self._get_obs(), float(reward), terminated, truncated, {"pos_err": pos_err}
