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


def run_swarm_formation():
    print("=" * 70)
    print("MULTI-AGENT ROBOTICS: 3-UAV DECENTRALIZED FORMATION FLIGHT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_quadrotor_policy.pth")

    agent = ActorCritic(obs_dim=12, action_dim=4).to(device)
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    agent.eval()

    num_uavs = 3
    uavs = [QuadrotorDynamics() for _ in range(num_uavs)]

    # Initial spawns
    init_positions = [
        np.array([-1.0, -0.8, 1.0]),
        np.array([0.0, -1.0, 1.0]),
        np.array([1.0, -0.8, 1.0])
    ]

    for i in range(num_uavs):
        uavs[i].reset(init_pos=init_positions[i], init_vel=np.zeros(3))

    # Shared formation offset relative to moving swarm centroid
    formation_offsets = [
        np.array([0.0, 0.4, 0.0]),    # Lead drone
        np.array([-0.6, -0.4, 0.0]),  # Left wing
        np.array([0.6, -0.4, 0.0])    # Right wing
    ]

    dt = 0.02
    steps = 400
    trajectories = [[] for _ in range(num_uavs)]

    for step in range(steps):
        t = step * dt
        # Centroid travels along a smooth path
        centroid_target = np.array([0.8 * np.sin(0.5 * t), 0.5 * t - 1.0, 1.5])

        for i in range(num_uavs):
            target_i = centroid_target + formation_offsets[i]
            
            # Inter-agent collision avoidance
            repulsion = np.zeros(3)
            for j in range(num_uavs):
                if i != j:
                    d_vec = uavs[i].pos - uavs[j].pos
                    dist = np.linalg.norm(d_vec)
                    if dist < 0.5:
                        repulsion += (d_vec / (dist + 1e-6)) * (0.5 - dist) * 1.5

            effective_target = target_i + repulsion
            pos_err = effective_target - uavs[i].pos
            obs = np.concatenate([pos_err, uavs[i].vel, uavs[i].quat[1:4], uavs[i].omega], dtype=np.float32)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                act = agent.actor_mean(obs_t).squeeze(0).cpu().numpy()

            norm_act = np.clip((act + 1.0) / 2.0, 0.0, 1.0)
            state = uavs[i].step(norm_act, dt=dt)
            trajectories[i].append(state["pos"].copy())

    trajectories = [np.array(tr) for tr in trajectories]

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    colors = ["crimson", "seagreen", "navy"]
    labels = ["Leader (UAV 1)", "Wingman Left (UAV 2)", "Wingman Right (UAV 3)"]

    for i in range(num_uavs):
        ax.plot(trajectories[i][:, 0], trajectories[i][:, 1], trajectories[i][:, 2], color=colors[i], lw=2, label=labels[i])
        ax.scatter([trajectories[i][-1, 0]], [trajectories[i][-1, 1]], [trajectories[i][-1, 2]], color=colors[i], s=60)

    ax.set_title("3-UAV Synchronized Triangle Formation Flight")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    ax.view_init(elev=30, azim=40)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "multi_uav_swarm_formation.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT SAVED] Swarm formation plot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    run_swarm_formation()