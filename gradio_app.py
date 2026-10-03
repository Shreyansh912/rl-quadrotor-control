import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

sys.path.append(os.path.dirname(__file__))
from dynamics.quadrotor_dynamics import QuadrotorDynamics
from rl_controller.ppo_network import ActorCritic

# Load trained PPO agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = os.path.join(os.path.dirname(__file__), "results", "ppo_quadrotor_policy.pth")
agent = ActorCritic(obs_dim=12, action_dim=4).to(device)
if os.path.exists(weights_path):
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    agent.eval()


def simulate_flight(target_z, payload_mass, wind_gust, kill_motor):
    dyn = QuadrotorDynamics()
    dyn.mass = 0.8 + float(payload_mass)
    dyn.reset(init_pos=np.array([0.0, 0.0, target_z]), init_vel=np.zeros(3))

    target_pos = np.array([0.0, 0.0, target_z])
    dt = 0.02
    steps = 300

    time_hist, pos_hist, throttle_hist = [], [], []

    for step in range(steps):
        t = step * dt
        pos_err = target_pos - dyn.pos
        obs = np.concatenate([pos_err, dyn.vel, dyn.quat[1:4], dyn.omega], dtype=np.float32)
        obs_t = torch.tensor(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            act = agent.actor_mean(obs_t).squeeze(0).cpu().numpy()

        norm_act = np.clip((act + 1.0) / 2.0, 0.0, 1.0)

        # Motor cutout at t >= 1.5s
        if t >= 1.5:
            if kill_motor == "Motor 1": norm_act[0] = 0.0
            elif kill_motor == "Motor 2": norm_act[1] = 0.0
            elif kill_motor == "Motor 3": norm_act[2] = 0.0
            elif kill_motor == "Motor 4": norm_act[3] = 0.0

        wind_force = np.random.normal(0, float(wind_gust) * 0.25, size=3)
        state = dyn.step(norm_act, dt=dt, external_force=wind_force)

        time_hist.append(t)
        pos_hist.append(state["pos"].copy())
        throttle_hist.append(norm_act.copy())

    pos_hist = np.array(pos_hist)
    throttle_hist = np.array(throttle_hist)
    time_hist = np.array(time_hist)

    # Figure 1: 3D Coordinate Tracking
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(time_hist, pos_hist[:, 0], label="X (m)", color="crimson")
    ax1.plot(time_hist, pos_hist[:, 1], label="Y (m)", color="seagreen")
    ax1.plot(time_hist, pos_hist[:, 2], label="Z (m)", color="navy", lw=2)
    ax1.axhline(target_z, color="black", linestyle="--", label=f"Target Z ({target_z}m)")
    if kill_motor != "None":
        ax1.axvline(1.5, color="red", linestyle=":", label="Failure Injected")
    ax1.set_ylabel("Position (m)")
    ax1.set_title("3D Trajectory Tracking")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Figure 2: Individual Throttle Commands
    rotor_colors = ["royalblue", "darkorange", "forestgreen", "firebrick"]
    for i in range(4):
        ax2.plot(time_hist, throttle_hist[:, i], label=f"Motor {i+1}", color=rotor_colors[i])
    if kill_motor != "None":
        ax2.axvline(1.5, color="red", linestyle=":", label="Failure Injected")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Throttle [0, 1]")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()

    final_err = np.linalg.norm(pos_hist[-1] - target_pos)
    rmse = np.sqrt(np.mean(np.linalg.norm(pos_hist - target_pos, axis=1) ** 2))
    telemetry_summary = f"Terminal Tracking Error: {final_err:.4f} m | Flight RMSE: {rmse:.4f} m"

    return fig, telemetry_summary


demo = gr.Interface(
    fn=simulate_flight,
    inputs=[
        gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="Target Altitude (m)"),
        gr.Slider(0.0, 0.5, value=0.0, step=0.05, label="Payload Mass Drift (kg)"),
        gr.Slider(0.0, 8.0, value=2.0, step=0.5, label="Wind Gust Speed (m/s)"),
        gr.Dropdown(["None", "Motor 1", "Motor 2", "Motor 3", "Motor 4"], value="None", label="Motor Cutout Failure")
    ],
    outputs=[
        gr.Plot(label="Live Flight Telemetry"),
        gr.Textbox(label="Benchmark Metrics")
    ],
    title="🚁 6-DOF Quadrotor Neural Flight Control Dashboard",
    description="Interactive disturbance rejection & hardware fault-tolerance testing powered by PyTorch continuous PPO."
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)