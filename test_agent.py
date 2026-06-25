import gymnasium as gym
import time
from stable_baselines3 import PPO
from minigrid.wrappers import ImgObsWrapper
from env.factory_escape_env import FactoryEscapeEnv

def main():
    print("Loading trained PPO agent...")
    # 1. Create the environment with human rendering enabled
    env = FactoryEscapeEnv(size=16, render_mode="human")
    
    # 2. Wrap it so the observation matches what the CNN expects
    env = ImgObsWrapper(env)
    
    # 3. Load the saved model
    model = PPO.load("ppo_factory_escape")

    # 4. Run 3 test episodes
    for ep in range(3):
        obs, info = env.reset()
        done = False
        truncated = False
        step_count = 0
        
        print(f"--- Episode {ep + 1} ---")
        while not done and not truncated:
            # The model predicts the best action based on the current grid visual
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            step_count += 1
            # Slow down the render frame rate so we can watch it
            time.sleep(0.1)
            
            if done or truncated:
                status = "Success!" if info.get('success') else "Failure"
                print(f"Episode ended after {step_count} steps. Result: {status}")
                time.sleep(1) # Pause briefly before starting the next episode

    env.close()

    print("\nNote: The agent trained for 100k steps. To achieve a 100% success rate")
    print("saving all workers, it will likely need to train for 1M - 2M steps.")

if __name__ == "__main__":
    main()
