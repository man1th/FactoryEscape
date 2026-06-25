import gymnasium as gym
import numpy as np
from env.factory_escape_env import FactoryEscapeEnv

def main():
    print("Initializing Random Baseline Agent...")
    env = FactoryEscapeEnv(size=16, render_mode=None)
    
    episodes = 100
    successes = 0
    total_workers_saved = []
    total_rewards = []
    total_steps = []
    
    print(f"Running {episodes} episodes. Please wait...")
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        steps = 0
        
        while not done and not truncated:
            # Take a completely random action from the action space
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            
        # Tally metrics
        if info.get('success'):
            successes += 1
        total_workers_saved.append(env.unwrapped.saved_workers)
        total_rewards.append(ep_reward)
        total_steps.append(steps)
        
    # Print the evaluation metrics
    print("\n" + "="*35)
    print("    RANDOM AGENT BASELINE RESULTS")
    print("    (Evaluated over 100 episodes)")
    print("="*35)
    print(f"Success Rate           : {successes/episodes * 100:.1f}%")
    print(f"Average Workers Saved  : {np.mean(total_workers_saved):.2f} out of {env.unwrapped.n_workers}")
    print(f"Average Reward         : {np.mean(total_rewards):.2f}")
    print(f"Average Episode Length : {np.mean(total_steps):.1f} steps")
    print("="*35)

if __name__ == "__main__":
    main()
