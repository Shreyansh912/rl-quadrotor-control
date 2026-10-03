import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from envs.quadrotor_trajectory_env import QuadrotorTrajectoryEnv
from rl_controller.ppo_network import ActorCritic


def evaluate_trajectory_tracking():
    print("=" * 70)
    print("EVALUATING DYNAMIC 3D TRAJECTORY TRACKING (FIGURE-8 LEMNISCATE)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_trajectory_policy.pth")

    agent = ActorCritic(obs_dim=15, action_dim=4).to(device)
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    agent.eval()

    env = QuadrotorTrajectoryEnv(max_steps=500)
    obs, _ = env.reset()

    traj_actual = []
    traj_ref = []
    time_log = []

    print("[*] Running 500-step dynamic trajectory tracking test...")

    for step in range(500):
        t = step * env.dt
        p_ref, _ = env.get_reference_trajectory(t)

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = agent.actor_mean(obs_tensor).squeeze(0).cpu().numpy()

        obs, _, terminated, truncated, _ = env.step(action)

        time_log.append(t)
        traj_actual.append(env.dynamics.pos.copy())
        traj_ref.append(p_ref.copy())

        if terminated or truncated:
            break

    traj_actual = np.array(traj_actual)
    traj_ref = np.array(traj_ref)
    time_log = np.array(time_log)

    pos_errors = np.linalg.norm(traj_actual - traj_ref, axis=1)
    rmse = np.sqrt(np.mean(pos_errors ** 2))
    max_err = np.max(pos_errors)

    print(f"\n[BENCHMARK RESULTS]")
    print(f"  * 3D Trajectory Tracking RMSE: {rmse:.4f} m")
    print(f"  * Maximum Tracking Deviation:  {max_err:.4f} m")

    # 1. 3D Trajectory Plot
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(traj_ref[:, 0], traj_ref[:, 1], traj_ref[:, 2], "k--", lw=2, label="Reference Path (Lemniscate)")
    ax1.plot(traj_actual[:, 0], traj_actual[:, 1], traj_actual[:, 2], "crimson", lw=2.5, label="PPO Flight Track")
    ax1.scatter([traj_actual[0, 0]], [traj_actual[0, 1]], [traj_actual[0, 2]], color="green", s=50, label="Start")
    ax1.set_title("3D Figure-8 Flight Trajectory")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.legend()
    ax1.view_init(elev=25, azim=45)

    # 2. Time-series Axis Tracking
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(time_log, traj_ref[:, 0], "r--", alpha=0.6, label="X Ref")
    ax2.plot(time_log, traj_actual[:, 0], "r", label="X Actual")
    ax2.plot(time_log, traj_ref[:, 1], "g--", alpha=0.6, label="Y Ref")
    ax2.plot(time_log, traj_actual[:, 1], "g", label="Y Actual")
    ax2.plot(time_log, traj_ref[:, 2], "b--", alpha=0.6, label="Z Ref")
    ax2.plot(time_log, traj_actual[:, 2], "b", label="Z Actual")
    ax2.set_title("Coordinate Tracking Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_figure8_tracking.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT SAVED] Trajectory tracking plot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    evaluate_trajectory_tracking()