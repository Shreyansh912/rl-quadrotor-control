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


def run_energy_smoothness_benchmark():
    print("=" * 70)
    print("AEROSPACE METRIC: MOTOR CHATTER & ELECTRICAL POWER DRAW ANALYSIS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_quadrotor_policy.pth")

    agent = ActorCritic(obs_dim=12, action_dim=4).to(device)
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    agent.eval()

    dyn = QuadrotorDynamics()
    dyn.reset(init_pos=np.array([0.0, 0.0, 1.5]), init_vel=np.zeros(3))
    target_pos = np.array([0.0, 0.0, 1.5])
    dt = 0.02
    steps = 300

    time_hist, power_hist, delta_act_hist = [], [], []
    prev_act = np.zeros(4)

    for step in range(steps):
        t = step * dt
        pos_err = target_pos - dyn.pos
        obs = np.concatenate([pos_err, dyn.vel, dyn.quat[1:4], dyn.omega], dtype=np.float32)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            act = agent.actor_mean(obs_t).squeeze(0).cpu().numpy()

        norm_act = np.clip((act + 1.0) / 2.0, 0.0, 1.0)
        
        # Power model: P proportional to sum of motor RPM^3
        # Baseline hover power is ~180W on a 4S LiPo setup
        power_draw = 180.0 * np.sum(norm_act ** 3) / (4.0 * (0.33 ** 3))
        act_delta = np.linalg.norm(norm_act - prev_act) / dt

        state = dyn.step(norm_act, dt=dt)
        prev_act = norm_act.copy()

        time_hist.append(t)
        power_hist.append(power_draw)
        delta_act_hist.append(act_delta)

    time_hist = np.array(time_hist)
    power_hist = np.array(power_hist)
    delta_act_hist = np.array(delta_act_hist)

    avg_power = np.mean(power_hist[50:])  # steady-state average
    total_energy_kj = np.trapz(power_hist, time_hist) / 1000.0
    mean_chatter = np.mean(delta_act_hist)

    print("\n[AEROSPACE EFFICIENCY METRICS]")
    print(f"  * Average Steady-State Power Draw: {avg_power:.2f} W")
    print(f"  * Cumulative Energy Consumed:      {total_energy_kj:.2f} kJ")
    print(f"  * Action Slew Rate (Chatter RMS):  {mean_chatter:.2f} s^-1")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(time_hist, power_hist, color="firebrick", lw=2)
    ax1.axhline(avg_power, color="black", linestyle="--", label=f"Mean Steady-State Power ({avg_power:.1f}W)")
    ax1.set_ylabel("Power Draw (W)")
    ax1.set_title("Dynamic Electrical Power Demand & Control Rate Smoothness")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(time_hist, delta_act_hist, color="steelblue", lw=1.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Action Slew Rate ||da/dt||")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "energy_efficiency_telemetry.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT SAVED] Energy telemetry plot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    run_energy_smoothness_benchmark()