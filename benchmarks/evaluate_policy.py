import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from envs.quadrotor_hover_env import QuadrotorHoverEnv
from rl_controller.ppo_network import ActorCritic


def quat_to_euler(q):
    """Converts unit quaternion [qw, qx, qy, qz] to Euler angles (roll, pitch, yaw) in deg."""
    qw, qx, qy, qz = q
    # Roll (x-axis)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    # Yaw (z-axis)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])


def run_evaluation():
    print("=" * 70)
    print("EVALUATING TRAINED PPO POLICY ON 6-DOF QUADROTOR FLIGHT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_quadrotor_policy.pth")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    # Load agent
    agent = ActorCritic().to(device)
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    agent.eval()

    env = QuadrotorHoverEnv(max_steps=400)
    obs, _ = env.reset()

    # Telemetry logging buffers
    time_log = []
    pos_log = []
    vel_log = []
    euler_log = []
    actions_log = []

    print("[*] Simulating closed-loop hover response for 400 steps (8.0 seconds)...")

    for step in range(400):
        t = step * env.dt
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # Use deterministic mean action for evaluation
            action_mean = agent.actor_mean(obs_tensor)
            action = action_mean.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, _ = env.step(action)

        time_log.append(t)
        pos_log.append(env.dynamics.pos.copy())
        vel_log.append(env.dynamics.vel.copy())
        euler_log.append(quat_to_euler(env.dynamics.quat))
        actions_log.append(np.clip((action + 1.0) / 2.0, 0.0, 1.0))

        if terminated or truncated:
            break

    time_log = np.array(time_log)
    pos_log = np.array(pos_log)
    vel_log = np.array(vel_log)
    euler_log = np.array(euler_log)
    actions_log = np.array(actions_log)

    final_err = np.linalg.norm(env.target_pos - pos_log[-1])
    print(f"[METRIC] Final Position Error: {final_err:.4f} m (Target: [0, 0, 1.5])")
    print(f"[METRIC] Final Steady-State Altitude: {pos_log[-1, 2]:.4f} m")

    # Generate 4-Panel Telemetry Figure
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    # 1. Position Trajectory
    axs[0, 0].plot(time_log, pos_log[:, 0], label="x (m)", color="crimson")
    axs[0, 0].plot(time_log, pos_log[:, 1], label="y (m)", color="seagreen")
    axs[0, 0].plot(time_log, pos_log[:, 2], label="z (m)", color="navy")
    axs[0, 0].axhline(1.5, color="black", linestyle="--", alpha=0.7, label="Target Z (1.5m)")
    axs[0, 0].set_title("Position Response ($p_x, p_y, p_z$)")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Position (m)")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    axs[0, 0].legend()

    # 2. Attitude Euler Angles
    axs[0, 1].plot(time_log, euler_log[:, 0], label="Roll (deg)", color="purple")
    axs[0, 1].plot(time_log, euler_log[:, 1], label="Pitch (deg)", color="darkorange")
    axs[0, 1].plot(time_log, euler_log[:, 2], label="Yaw (deg)", color="teal")
    axs[0, 1].set_title("Attitude Stability (Euler Angles)")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Angle (°)")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    axs[0, 1].legend()

    # 3. Linear Velocities
    axs[1, 0].plot(time_log, vel_log[:, 0], label="v_x (m/s)", color="crimson")
    axs[1, 0].plot(time_log, vel_log[:, 1], label="v_y (m/s)", color="seagreen")
    axs[1, 0].plot(time_log, vel_log[:, 2], label="v_z (m/s)", color="navy")
    axs[1, 0].set_title("Linear Velocity Damping")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Velocity (m/s)")
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)
    axs[1, 0].legend()

    # 4. Normalized Motor Throttle Outputs
    for i in range(4):
        axs[1, 1].plot(time_log, actions_log[:, i], label=f"Motor {i+1} Throttle")
    axs[1, 1].set_title("Continuous Motor Throttle Commands")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Normalized Command [0, 1]")
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    axs[1, 1].legend()

    plt.tight_layout()
    out_plot = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_hover_telemetry.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"[PLOT SAVED] Telemetry plot saved to: {os.path.abspath(out_plot)}")


if __name__ == "__main__":
    run_evaluation()