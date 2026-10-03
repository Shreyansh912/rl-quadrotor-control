import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dynamics.quadrotor_dynamics import QuadrotorDynamics
from rl_controller.ppo_network import ActorCritic


class PIDFlightController:
    """Cascaded 3D PID Controller for Quadrotor Position and Attitude."""
    def __init__(self):
        # Position gains
        self.kp_pos = np.array([1.8, 1.8, 4.0])
        self.kd_pos = np.array([1.2, 1.2, 2.5])
        self.ki_pos = np.array([0.05, 0.05, 0.2])
        self.integral_err = np.zeros(3)

    def compute_action(self, pos, vel, target_pos, dt=0.02):
        err = target_pos - pos
        self.integral_err += err * dt
        self.integral_err = np.clip(self.integral_err, -1.0, 1.0)

        # Desired acceleration
        acc_des = self.kp_pos * err - self.kd_pos * vel + self.ki_pos * self.integral_err
        # Nominal thrust for hover ~ 0.33
        base_throttle = 0.33 + acc_des[2] * 0.05
        # Differential pitch/roll adjustments
        t1 = base_throttle - acc_des[0] * 0.04 + acc_des[1] * 0.04
        t2 = base_throttle + acc_des[0] * 0.04 - acc_des[1] * 0.04
        t3 = base_throttle + acc_des[0] * 0.04 + acc_des[1] * 0.04
        t4 = base_throttle - acc_des[0] * 0.04 - acc_des[1] * 0.04

        action = np.clip([t1, t2, t3, t4], 0.0, 1.0)
        return action * 2.0 - 1.0  # Normalized to [-1, 1]


def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: PPO NEURAL CONTROLLER VS CLASSICAL PID (WIND TURBULENCE)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_quadrotor_policy.pth")

    agent = ActorCritic().to(device)
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    agent.eval()

    pid = PIDFlightController()
    target_pos = np.array([0.0, 0.0, 1.5])
    sim_steps = 300
    dt = 0.02

    np.random.seed(42)
    # Wind disturbance force vector over time (turbulent gusts)
    wind_forces = np.random.normal(loc=0.0, scale=1.2, size=(sim_steps, 3))
    wind_forces[:, 2] *= 0.5

    # 1. Simulate PPO
    dyn_ppo = QuadrotorDynamics()
    dyn_ppo.reset(init_pos=[0.2, -0.2, 1.3])
    ppo_pos = []

    for step in range(sim_steps):
        pos_err = target_pos - dyn_ppo.pos
        obs = np.concatenate([pos_err, dyn_ppo.vel, dyn_ppo.quat[1:4], dyn_ppo.omega], dtype=np.float32)
        obs_t = torch.tensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            act = agent.actor_mean(obs_t).squeeze(0).cpu().numpy()
        norm_act = np.clip((act + 1.0) / 2.0, 0.0, 1.0)
        state = dyn_ppo.step(norm_act, dt=dt, external_force=wind_forces[step])
        ppo_pos.append(state["pos"].copy())

    # 2. Simulate PID
    dyn_pid = QuadrotorDynamics()
    dyn_pid.reset(init_pos=[0.2, -0.2, 1.3])
    pid_pos = []

    for step in range(sim_steps):
        act = pid.compute_action(dyn_pid.pos, dyn_pid.vel, target_pos, dt=dt)
        norm_act = np.clip((act + 1.0) / 2.0, 0.0, 1.0)
        state = dyn_pid.step(norm_act, dt=dt, external_force=wind_forces[step])
        pid_pos.append(state["pos"].copy())

    ppo_pos = np.array(ppo_pos)
    pid_pos = np.array(pid_pos)
    time_axis = np.arange(sim_steps) * dt

    ppo_rmse = np.sqrt(np.mean(np.sum((ppo_pos - target_pos) ** 2, axis=1)))
    pid_rmse = np.sqrt(np.mean(np.sum((pid_pos - target_pos) ** 2, axis=1)))

    print(f"\n[BENCHMARK RESULT]")
    print(f"  * PPO Controller Tracking RMSE:      {ppo_rmse:.4f} m")
    print(f"  * Cascaded PID Controller Tracking RMSE: {pid_rmse:.4f} m")
    print(f"  * Relative Improvement:              {((pid_rmse - ppo_rmse) / pid_rmse) * 100:.2f}%")

    # Plot Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, ppo_pos[:, 2], color="crimson", lw=2.2, label=f"PPO Policy (RMSE: {ppo_rmse:.3f}m)")
    plt.plot(time_axis, pid_pos[:, 2], color="navy", linestyle="--", lw=2.0, label=f"Classical PID (RMSE: {pid_rmse:.3f}m)")
    plt.axhline(1.5, color="green", linestyle=":", lw=1.8, label="Target Z (1.5m)")
    plt.title("Hover Altitude Tracking Under Stochastic Wind Turbulence")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude Z (m)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_plot = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_vs_pid_benchmark.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"[PLOT SAVED] Benchmark comparison saved to: {os.path.abspath(out_plot)}")


if __name__ == "__main__":
    run_benchmark()