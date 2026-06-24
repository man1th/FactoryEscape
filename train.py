import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.factory_escape_env import FactoryEscapeEnv

# 1. Wrap the environment
# MiniGrid environments provide 'image' observations which are perfect for CNNs.
def make_env():
    return FactoryEscapeEnv(size=16, render_mode=None)

# 2. Vectorize the environment (runs multiple copies for faster training)
env = make_vec_env(make_env, n_envs=4)

# 3. Define the PPO agent
# We use 'CnnPolicy' because the state is an image grid
model = PPO("CnnPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048)

print("Starting training...")
# 4. Train the agent
model.learn(total_timesteps=100000)

# 5. Save the agent
model.save("ppo_factory_escape")
print("Training complete. Model saved as ppo_factory_escape.zip")
