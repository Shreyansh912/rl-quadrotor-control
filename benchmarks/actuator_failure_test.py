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


def run_motor_failure_benchmark():
    print("=" * 70)
    print("STRESS TEST: CATASTROPHIC MOTOR 1 FAILURE AT t = 2.0s")
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
    total_steps = 300  # 6.0 seconds

    time_log, pos_log, throttle_log = [], [], []

    for step in range(total_steps):
        t = step * dt

        # State observation
        pos_err = target_pos - dyn.pos
        obs = np.concatenate([pos_err, dyn.vel, dyn.quat[1:4], dyn.omega], dtype=np.float32)
        obs_t = torch.tensor(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            act = agent.actor_mean(obs_t).squeeze(0).cpu().numpy()

        norm_act = np.clip((act + 1.0) / 2.0, 0.0, 1.0)

        # Inject 100% loss of Motor 1 at t >= 2.0s
        if t >= 2.0:
            norm_act[0] = 0.0  # Rotor 1 dead

        state = dyn.step(norm_act, dt=dt)

        time_log.append(t)
        pos_log.append(state["pos"].copy())
        throttle_log.append(norm_act.copy())

    pos_log = np.array(pos_log)
    throttle_log = np.array(throttle_log)
    time_log = np.array(time_log)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(time_log, pos_log[:, 0], label="x (m)", color="crimson")
    ax1.plot(time_log, pos_log[:, 1], label="y (m)", color="seagreen")
    ax1.plot(time_log, pos_log[:, 2], label="z (m)", color="navy", lw=2)
    ax1.axvline(2.0, color="black", linestyle="--", label="Motor 1 Failure Event")
    ax1.set_ylabel("Position (m)")
    ax1.set_title("Fail-Safe Response Under Sudden 100% Motor 1 Cutout")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    ax2.plot(time_log, throttle_log[:, 0], label="Motor 1 (Failed)", color="black", lw=2)
    ax2.plot(time_log, throttle_log[:, 1], label="Motor 2", color="royalblue")
    ax2.plot(time_log, throttle_log[:, 2], label="Motor 3", color="darkorange")
    ax2.plot(time_log, throttle_log[:, 3], label="Motor 4", color="purple")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Rotor Throttle [0, 1]")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "actuator_failure_response.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Actuator failure benchmark plot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    run_motor_failure_benchmark()