import os
import sys
import time
import numpy as np
import pyvista as pv
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from envs.quadrotor_hover_env import QuadrotorHoverEnv
from rl_controller.ppo_network import ActorCritic


def build_quadrotor_mesh(center=(0, 0, 0), arm_length=0.22):
    """Constructs a composite 3D geometric mesh of a quadrotor airframe."""
    x, y, z = center
    d = arm_length / np.sqrt(2.0)

    # Fuselage hub
    body = pv.Cylinder(center=(x, y, z), direction=(0, 0, 1), radius=0.065, height=0.035)

    # Structural X-Arms
    arm1 = pv.Cylinder(center=(x, y, z), direction=(1, 1, 0), radius=0.010, height=arm_length * 2)
    arm2 = pv.Cylinder(center=(x, y, z), direction=(1, -1, 0), radius=0.010, height=arm_length * 2)

    # Rotor discs
    rotor_offsets = [(d, d), (-d, -d), (d, -d), (-d, d)]
    rotors = []
    for ox, oy in rotor_offsets:
        motor = pv.Cylinder(center=(x + ox, y + oy, z + 0.018), direction=(0, 0, 1), radius=0.018, height=0.025)
        disk = pv.Cylinder(center=(x + ox, y + oy, z + 0.03), direction=(0, 0, 1), radius=0.08, height=0.004)
        rotors.extend([motor, disk])

    return body.merge([arm1, arm2] + rotors)


def run_3d_flight_visualization():
    print("=" * 70)
    print("3D REINFORCEMENT LEARNING FLIGHT TRAJECTORY VIEWER")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_quadrotor_policy.pth")

    agent = ActorCritic().to(device)
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    agent.eval()

    env = QuadrotorHoverEnv(max_steps=250)
    obs, _ = env.reset()

    traj_positions = []
    print("[*] Generating trajectory using trained PPO policy...")

    for _ in range(250):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = agent.actor_mean(obs_tensor).squeeze(0).cpu().numpy()
        obs, _, terminated, truncated, _ = env.step(action)
        traj_positions.append(env.dynamics.pos.copy())
        if terminated or truncated:
            break

    traj_positions = np.array(traj_positions)

    # Setup PyVista Scene
    plotter = pv.Plotter(title="PPO Quadrotor Flight - 3D VTK Viewer", window_size=(1024, 768))
    plotter.set_background("#18191c")

    # Target waypoint marker (sphere)
    target_sphere = pv.Sphere(radius=0.06, center=[0.0, 0.0, 1.5])
    plotter.add_mesh(target_sphere, color="#00ffcc", opacity=0.85, label="Target Hover [0, 0, 1.5]")

    # Target indicator ring
    target_ring = pv.Disc(center=[0.0, 0.0, 1.5], inner=0.15, outer=0.20, normal=(0, 0, 1))
    plotter.add_mesh(target_ring, color="#00ffcc", opacity=0.4)

    # Ground Plane
    ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=4.0, j_size=4.0)
    plotter.add_mesh(ground, color="#2b2d30", show_edges=True, edge_color="#3c3f41", opacity=0.8)

    # Spline trajectory
    spline = pv.Spline(traj_positions, 300)
    plotter.add_mesh(spline, color="#ff4757", line_width=3, label="PPO Flight Path")

    # Initial drone actor
    drone_mesh = build_quadrotor_mesh(center=traj_positions[0])
    drone_actor = plotter.add_mesh(drone_mesh, color="#3742fa")

    plotter.camera_position = [(-2.8, -2.8, 2.5), (0.0, 0.0, 1.2), (0, 0, 1)]
    plotter.add_axes()
    plotter.add_legend(bcolor=None)

    plotter.show(auto_close=False, interactive_update=True)

    # Animate flight
    for pos in traj_positions:
        updated = build_quadrotor_mesh(center=pos)
        drone_actor.mapper.SetInputData(updated)
        plotter.update()
        time.sleep(0.02)

    # Export snapshot
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    snapshot_path = os.path.join(out_dir, "ppo_3d_flight_render.png")
    plotter.screenshot(snapshot_path)
    print(f"\n[RENDER SAVED] 3D flight trajectory saved to: {os.path.abspath(snapshot_path)}")

    plotter.show(interactive=True)


if __name__ == "__main__":
    run_3d_flight_visualization()