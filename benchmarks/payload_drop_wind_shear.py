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


def wind_shear_profile(altitude, v_ref=3.5, z_ref=1.5, z0=0.05):
    """Logarithmic atmospheric wind shear profile."""
    z = max(altitude, 0.06)
    return v_ref * (np.log(z / z0) / np.log(z_ref / z0))


def run_payload_windshear_simulation():
    print("=" * 70)
    print("AEROSPACE TEST: ATMOSPHERIC WIND SHEAR + MID-AIR PAYLOAD DROP")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_quadrotor_policy.pth")

    agent = ActorCritic(obs_dim=12, action_dim=4).to(device)
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    agent.eval()

    dyn = QuadrotorDynamics()
    # Initial drone mass with payload attached: 0.8 + 0.25 = 1.05 kg
    dyn.mass = 1.05
    dyn.reset(init_pos=np.array([0.0, 0.0, 1.5]), init_vel=np.zeros(3))

    target_pos = np.array([0.0, 0.0, 1.5])
    dt = 0.02
    total_steps = 350

    time_log, z_log, wind_log, mass_log = [], [], [], []

    for step in range(total_steps):
        t = step * dt

        # Payload drop event at t = 2.5s (instant mass loss of 250g)
        if t >= 2.5 and dyn.mass > 0.80:
            dyn.mass = 0.80

        # Dynamic shear wind vector
        shear_vx = wind_shear_profile(dyn.pos[2])
        gust_force = np.array([shear_vx + np.random.normal(0, 0.3), np.random.normal(0, 0.2), 0.0])

        pos_err = target_pos - dyn.pos
        obs = np.concatenate([pos_err, dyn.vel, dyn.quat[1:4], dyn.omega], dtype=np.float32)
        obs_t = torch.tensor(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            act = agent.actor_mean(obs_t).squeeze(0).cpu().numpy()

        norm_act = np.clip((act + 1.0) / 2.0, 0.0, 1.0)
        state = dyn.step(norm_act, dt=dt, external_force=gust_force)

        time_log.append(t)
        z_log.append(state["pos"][2])
        wind_log.append(shear_vx)
        mass_log.append(dyn.mass)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    ax1.plot(time_log, z_log, label="Altitude Z (m)", color="teal", lw=2)
    ax1.axhline(1.5, color="black", linestyle="--", alpha=0.7, label="Target Altitude (1.5m)")
    ax1.axvline(2.5, color="crimson", linestyle=":", label="Payload Drop Event (-250g)")
    ax1.set_ylabel("Altitude (m)")
    ax1.set_title("Altitude Regulation Under Atmospheric Wind Shear & Sudden Payload Release")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    ax2.plot(time_log, mass_log, label="Airframe Mass (kg)", color="purple", lw=2)
    ax2.plot(time_log, np.array(wind_log) * 0.1, label="Horizontal Wind Shear Force (x0.1 N)", color="orange", linestyle="--")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Mass / Force")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "wind_shear_payload_drop.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Payload release & wind shear plot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    run_payload_windshear_simulation()