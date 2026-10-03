import numpy as np
from envs.quadrotor_hover_env import QuadrotorHoverEnv

def main():
    print("Testing QuadrotorHoverEnv...")
    env = QuadrotorHoverEnv()
    obs, _ = env.reset()
    print(f"[INIT] Observation Shape: {obs.shape} (Expected: 12)")
    print(f"[INIT] Sample Observation:\n{obs}")

    total_reward = 0.0
    for step in range(100):
        # Sample random continuous action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"[RESET] Episode ended at step {step + 1} | Pos Error: {info['pos_err']:.2f}m")
            obs, _ = env.reset()

    print(f"[SUCCESS] 100 environmental steps executed cleanly. Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()