from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from env import TradingEnv
import pandas as pd
from stable_baselines3.common.env_util import make_vec_env

# Load the trained model
model = PPO.load("ppo_trading_agent")

df = pd.read_csv('nse_indexes.csv')
# df = pd.read_csv('test_indexes.csv')

# Initialize the environment
env = TradingEnv(df, 60)

vec_env = make_vec_env(lambda: env, n_envs=1)
# Evaluate the trained model
# mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# You may further want to visualize the performance using the agent 
# by simulating it on the environment

obs = vec_env.reset()
for i in range(6531-60):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    print(f'Step: {env.current_step}, Action: {"Buy" if action==1 else "Sell" if action==2 else "Hold" }, balance: {env.balance}, Rewards: {rewards}, Date: {env.date}, Buy and Hold: {env.buy_and_hold * env.initial_balance}, done: {dones}')