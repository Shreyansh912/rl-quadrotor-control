import os
import sys
import numpy as np
import pyvista as pv
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from envs.quadrotor_hover_env import QuadrotorHoverEnv
from rl_controller.ppo_network import ActorCritic
from vision.visualize_rl_flight import build_quadrotor_mesh


def generate_flight_gif():
    print("[*] Simulating policy rollout for GIF export...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_quadrotor_policy.pth")

    agent = ActorCritic().to(device)
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    agent.eval()

    env = QuadrotorHoverEnv(max_steps=200)
    obs, _ = env.reset()

    traj_positions = []
    for _ in range(200):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = agent.actor_mean(obs_tensor).squeeze(0).cpu().numpy()
        obs, _, terminated, truncated, _ = env.step(action)
        traj_positions.append(env.dynamics.pos.copy())
        if terminated or truncated:
            break

    traj_positions = np.array(traj_positions)

    # Off-screen rendering plotter
    plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
    plotter.set_background("#18191c")

    target_sphere = pv.Sphere(radius=0.06, center=[0.0, 0.0, 1.5])
    plotter.add_mesh(target_sphere, color="#00ffcc", opacity=0.85)

    ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=3.5, j_size=3.5)
    plotter.add_mesh(ground, color="#2b2d30", show_edges=True, edge_color="#3c3f41", opacity=0.8)

    spline = pv.Spline(traj_positions, 200)
    plotter.add_mesh(spline, color="#ff4757", line_width=2.5)

    drone_mesh = build_quadrotor_mesh(center=traj_positions[0])
    drone_actor = plotter.add_mesh(drone_mesh, color="#3742fa")

    plotter.camera_position = [(-2.5, -2.5, 2.2), (0.0, 0.0, 1.2), (0, 0, 1)]

    out_gif = os.path.join(os.path.dirname(__file__), "..", "results", "flight_demo.gif")
    plotter.open_gif(out_gif, fps=25)

    # Animate every second frame for concise file size
    for pos in traj_positions[::2]:
        updated = build_quadrotor_mesh(center=pos)
        drone_actor.mapper.SetInputData(updated)
        plotter.write_frame()

    plotter.close()
    print(f"[SUCCESS] High-res flight GIF saved to: {os.path.abspath(out_gif)}")


if __name__ == "__main__":
    generate_flight_gif()