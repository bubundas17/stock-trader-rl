import gym
import pandas as pd
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from gym.utils import seeding

from env import TradingEnv

# Read from your CSV data
df = pd.read_csv('nse_indexes.csv')

# Initialize the environment
env = TradingEnv(df, 60)

vec_env = make_vec_env(lambda: env, n_envs=1)

# Initialize the PPO agent
# model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=dict(net_arch=[4096*2, 4096*2, 4096]))
# model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=dict(net_arch=[4096*2, 4096*2, 4096]))
model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=dict(net_arch=[4096*2, 4096]))

# Train the agent for 10000 steps
model.learn(total_timesteps=50000 * 16)

# Save the model
model.save("ppo_trading_agent")