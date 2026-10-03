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


class SimulatedArUcoTracker:
    """Simulates camera pose estimation with measurement noise and frame-rate latency."""
    def __init__(self, marker_ground_truth=np.array([0.0, 0.0, 1.5]), fps=30):
        self.marker_gt = marker_ground_truth
        self.dt_camera = 1.0 / fps
        self.last_update = 0.0
        self.last_estimate = np.copy(marker_ground_truth)

    def estimate_relative_vector(self, drone_pos, current_time):
        # Update measurement only at camera frame intervals
        if current_time - self.last_update >= self.dt_camera:
            noise = np.random.normal(0.0, 0.015, size=3)  # ~1.5cm vision noise
            self.last_estimate = (self.marker_gt - drone_pos) + noise
            self.last_update = current_time
        return self.last_estimate


def run_vision_rl_mission():
    print("=" * 70)
    print("MISSION: VISION-AIDED AUTONOMOUS HOVER/APPROACH (PERCEPTION + PPO)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_quadrotor_policy.pth")

    agent = ActorCritic().to(device)
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    agent.eval()

    tracker = SimulatedArUcoTracker(marker_ground_truth=np.array([0.0, 0.0, 1.5]))
    dyn = QuadrotorDynamics()
    # Spawn drone 1.2m offset horizontally and 0.5m lower
    dyn.reset(init_pos=np.array([0.8, -0.7, 1.0]), init_vel=np.zeros(3))

    sim_steps = 350
    dt = 0.02
    time_log = []
    pos_log = []

    print("[*] Running closed-loop vision + neural attitude control...")

    for step in range(sim_steps):
        t = step * dt
        # 1. Vision perception step (ArUco relative vector)
        pos_err_perceived = tracker.estimate_relative_vector(dyn.pos, t)

        # 2. State vector constructed with vision-derived position error
        obs = np.concatenate([
            pos_err_perceived,
            dyn.vel,
            dyn.quat[1:4],
            dyn.omega
        ], dtype=np.float32)

        obs_t = torch.tensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            act = agent.actor_mean(obs_t).squeeze(0).cpu().numpy()

        norm_act = np.clip((act + 1.0) / 2.0, 0.0, 1.0)
        state = dyn.step(norm_act, dt=dt)

        time_log.append(t)
        pos_log.append(state["pos"].copy())

    pos_log = np.array(pos_log)
    final_err = np.linalg.norm(pos_log[-1] - np.array([0.0, 0.0, 1.5]))
    print(f"\n[MISSION COMPLETE] Final Vision-Guided Offset: {final_err:.4f} m")

    # Save tracking plot
    plt.figure(figsize=(10, 4.5))
    plt.plot(time_log, pos_log[:, 0], label="x (Approach)", color="crimson")
    plt.plot(time_log, pos_log[:, 1], label="y (Approach)", color="seagreen")
    plt.plot(time_log, pos_log[:, 2], label="z (Ascent/Hold)", color="navy")
    plt.axhline(1.5, color="black", linestyle="--", alpha=0.6, label="Marker Target Z (1.5m)")
    plt.title("Vision-Guided Flight Trajectory (ArUco Perception -> PPO Motor Control)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "vision_rl_mission.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT SAVED] Mission telemetry saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    run_vision_rl_mission()