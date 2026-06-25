import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import ImgObsWrapper
from env.factory_escape_env import FactoryEscapeEnv

# 1. Custom CNN for MiniGrid's 7x7 observation space
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        # A smaller CNN designed specifically for 7x7 grids
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened output size automatically
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# 2. Environment builder
def make_env():
    env = FactoryEscapeEnv(size=16, render_mode=None)
    env = ImgObsWrapper(env)
    return env

if __name__ == "__main__":
    # 3. Vectorize the environment
    env = make_vec_env(make_env, n_envs=4)

    # 4. Inject the custom CNN into the PPO model via policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=2048
    )

    print("Starting PPO training...")
    # Train for 100,000 steps to test the pipeline (you will scale this to 1M later)
    model.learn(total_timesteps=100_000)

    model.save("ppo_factory_escape")
    print("Training complete. Model saved as ppo_factory_escape.zip")
