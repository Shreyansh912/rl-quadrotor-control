import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from envs.quadrotor_randomized_env import QuadrotorRandomizedEnv
from rl_controller.ppo_network import ActorCritic


def evaluate_robustness():
    print("=" * 70)
    print("SIM-TO-REAL STRESS TEST: DOMAIN RANDOMIZATION EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "results", "ppo_quadrotor_policy.pth")

    agent = ActorCritic(obs_dim=12, action_dim=4).to(device)
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    agent.eval()

    env = QuadrotorRandomizedEnv(max_steps=400)
    num_trials = 20
    errors = []
    masses = []

    print(f"[*] Running {num_trials} randomized Monte Carlo trials across physical parameter shifts...")

    for trial in range(num_trials):
        obs, _ = env.reset()
        trial_mass = env.dynamics.mass
        tau_val = getattr(env.dynamics, "tau", 0.03)
        masses.append(trial_mass)

        for _ in range(400):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                act = agent.actor_mean(obs_t).squeeze(0).cpu().numpy()
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break

        final_err = np.linalg.norm(env.dynamics.pos - env.target_pos)
        errors.append(final_err)
        print(f"  Trial {trial+1:02d} | Mass: {trial_mass:.3f} kg | Motor Lag: {tau_val*1000:.1f} ms | Final Error: {final_err:.4f} m")

    mean_err = np.mean(errors)
    max_err = np.max(errors)
    success_rate = (np.array(errors) < 0.30).mean() * 100.0

    print("\n[MONTE CARLO ROBUSTNESS SUMMARY]")
    print(f"  * Mean Steady-State Error: {mean_err:.4f} m")
    print(f"  * Worst-Case Deviation:    {max_err:.4f} m")
    print(f"  * Flight Stability Rate:   {success_rate:.1f}%")

    # Plot Distribution
    plt.figure(figsize=(9, 4.5))
    plt.scatter(masses, errors, color="crimson", s=60, edgecolors="black", alpha=0.8)
    plt.axhline(0.30, color="navy", linestyle="--", label="Acceptable Hover Threshold (0.30m)")
    plt.title("Policy Generalization Across Physical Mass & Actuator Uncertainty")
    plt.xlabel("Randomized Airframe Mass (kg)")
    plt.ylabel("Terminal Tracking Error (m)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_plot = os.path.join(os.path.dirname(__file__), "..", "results", "sim_to_real_robustness.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"[PLOT SAVED] Monte Carlo plot saved to: {os.path.abspath(out_plot)}")


if __name__ == "__main__":
    evaluate_robustness()
