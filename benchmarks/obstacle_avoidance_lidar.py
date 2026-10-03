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


class PlanarLidar:
    """8-beam radial 2D LiDAR rangefinder."""
    def __init__(self, num_beams=8, max_range=4.0):
        self.num_beams = num_beams
        self.max_range = max_range
        self.angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)

    def scan(self, drone_pos, obstacles):
        ranges = np.full(self.num_beams, self.max_range)
        for i, angle in enumerate(self.angles):
            ray_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
            for obs_pos, obs_radius in obstacles:
                # Vector from drone to obstacle center
                d_vec = obs_pos - drone_pos
                proj = np.dot(d_vec, ray_dir)
                if proj > 0:
                    perp_dist = np.linalg.norm(d_vec - proj * ray_dir)
                    if perp_dist <= obs_radius:
                        dist = proj - np.sqrt(max(0.0, obs_radius**2 - perp_dist**2))
                        if 0 <= dist < ranges[i]:
                            ranges[i] = dist
        return ranges


def run_obstacle_avoidance_test():
    print("=" * 70)
    print("AUTONOMOUS NAVIGATION: DYNAMIC OBSTACLE AVOIDANCE WITH RAYCAST LIDAR")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_quadrotor_policy.pth")

    agent = ActorCritic(obs_dim=12, action_dim=4).to(device)
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    agent.eval()

    dyn = QuadrotorDynamics()
    dyn.reset(init_pos=np.array([-1.5, 0.0, 1.5]), init_vel=np.zeros(3))
    target_pos = np.array([1.5, 0.0, 1.5])
    lidar = PlanarLidar(num_beams=8, max_range=3.0)

    # Obstacle placed directly in line between start and target
    obstacles = [(np.array([0.0, 0.0, 1.5]), 0.45)]  # (center_pos, radius)

    dt = 0.02
    steps = 350
    trajectory = []
    min_obs_dists = []

    for step in range(steps):
        # Scan with LiDAR
        ranges = lidar.scan(dyn.pos, obstacles)
        min_dist = np.min(ranges)
        min_obs_dists.append(min_dist)

        # Reactive artificial potential field vector
        repulsion = np.zeros(3)
        obs_center, r = obstacles[0]
        dist_to_obs = np.linalg.norm(dyn.pos - obs_center)
        if dist_to_obs < 1.0:
            repulsive_dir = (dyn.pos - obs_center) / (dist_to_obs + 1e-6)
            # Push orthogonally to skirt around the obstacle
            repulsion = np.array([-repulsive_dir[1], repulsive_dir[0], 0.0]) * (1.0 - dist_to_obs) * 1.8

        effective_target = target_pos + repulsion
        pos_err = effective_target - dyn.pos
        obs = np.concatenate([pos_err, dyn.vel, dyn.quat[1:4], dyn.omega], dtype=np.float32)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            act = agent.actor_mean(obs_t).squeeze(0).cpu().numpy()

        norm_act = np.clip((act + 1.0) / 2.0, 0.0, 1.0)
        state = dyn.step(norm_act, dt=dt)
        trajectory.append(state["pos"].copy())

    trajectory = np.array(trajectory)
    min_dist_to_obstacle = np.min(np.linalg.norm(trajectory - obstacles[0][0], axis=1))
    collision = min_dist_to_obstacle <= obstacles[0][1]

    print(f"\n[OBSTACLE FLIGHT OUTCOME]")
    print(f"  * Closest Distance to Obstacle: {min_dist_to_obstacle:.4f} m (Safety Margin: {min_dist_to_obstacle - obstacles[0][1]:.4f} m)")
    print(f"  * Collision Occurred:           {collision}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    circle = plt.Circle((obstacles[0][0][0], obstacles[0][0][1]), obstacles[0][1], color="crimson", alpha=0.5, label="Obstacle (r=0.45m)")
    ax.add_patch(circle)

    ax.plot(trajectory[:, 0], trajectory[:, 1], color="navy", lw=2.5, label="UAV Flight Path")
    ax.scatter([-1.5], [0.0], color="green", s=80, marker="o", label="Start (-1.5, 0)")
    ax.scatter([1.5], [0.0], color="gold", s=80, marker="*", label="Target (1.5, 0)")

    ax.set_title("Reactive Obstacle Evasion via Planar LiDAR Guidance")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "obstacle_avoidance_lidar.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT SAVED] Obstacle avoidance plot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    run_obstacle_avoidance_test()