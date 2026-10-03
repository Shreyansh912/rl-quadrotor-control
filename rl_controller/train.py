import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # Headless backend: eliminates all SVG/font warnings
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from envs.quadrotor_hover_env import QuadrotorHoverEnv
from rl_controller.ppo_network import ActorCritic


def train_ppo():
    print("=" * 70)
    print("STAGE 1: TRAINING PPO FLIGHT CONTROLLER (SIM-TO-REAL ENGINE)")
    print("=" * 70)

    learning_rate = 3e-4
    total_timesteps = 250000     # Sufficient exploration & convergence time
    num_steps = 1024
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.005
    vf_coef = 0.5
    max_grad_norm = 0.5
    batch_size = 64
    update_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0)})")

    env = QuadrotorHoverEnv()
    agent = ActorCritic().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    obs_buf = torch.zeros((num_steps, 12), dtype=torch.float32).to(device)
    actions_buf = torch.zeros((num_steps, 4), dtype=torch.float32).to(device)
    logprobs_buf = torch.zeros(num_steps, dtype=torch.float32).to(device)
    rewards_buf = torch.zeros(num_steps, dtype=torch.float32).to(device)
    dones_buf = torch.zeros(num_steps, dtype=torch.float32).to(device)
    values_buf = torch.zeros(num_steps, dtype=torch.float32).to(device)

    next_obs, _ = env.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(1, dtype=torch.float32).to(device)

    num_updates = total_timesteps // num_steps
    episode_rewards = []
    current_ep_reward = 0.0

    start_time = time.time()

    for update in range(1, num_updates + 1):
        for step in range(num_steps):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values_buf[step] = value.flatten()

            actions_buf[step] = action.squeeze(0)
            logprobs_buf[step] = logprob

            act_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(act_np)
            current_ep_reward += reward

            rewards_buf[step] = float(reward)
            done = terminated or truncated

            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                obs, _ = env.reset()

            next_obs = torch.tensor(obs, dtype=torch.float32).to(device)
            next_done = torch.tensor(float(done), dtype=torch.float32).to(device)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        b_obs = obs_buf
        b_actions = actions_buf
        b_logprobs = logprobs_buf
        b_advantages = advantages
        b_returns = returns

        b_inds = np.arange(num_steps)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) > 0 else 0.0
        fps = int((update * num_steps) / (time.time() - start_time))
        if update % 5 == 0 or update == num_updates:
            print(f"[PPO Update {update:3d}/{num_updates}] Step: {update * num_steps:6d} | "
                  f"Mean Reward: {avg_reward:6.2f} | Policy Loss: {pg_loss.item():.4f} | "
                  f"Value Loss: {v_loss.item():.4f} | FPS: {fps}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "ppo_quadrotor_policy.pth")
    torch.save(agent.state_dict(), weights_path)
    print(f"\n[MODEL SAVED] Trained policy weights saved to: {os.path.abspath(weights_path)}")

    if len(episode_rewards) > 0:
        plt.figure(figsize=(9, 4.5))
        plt.plot(episode_rewards, color="navy", alpha=0.25, label="Raw Episode Reward")
        window = min(20, len(episode_rewards))
        rolling = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window - 1, len(episode_rewards)), rolling, color="crimson", lw=2, label=f"{window}-Ep Moving Avg")
        plt.title("PPO Flight Controller Training Convergence")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        curve_path = os.path.join(out_dir, "ppo_training_convergence.png")
        plt.savefig(curve_path, dpi=300)
        plt.close()
        print(f"[PLOT SAVED] Learning curve saved to: {os.path.abspath(curve_path)}")


if __name__ == "__main__":
    train_ppo()