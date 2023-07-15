import gym
import pandas as pd
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from gym.utils import seeding

np.random.seed(0)

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, max_loss = 5000):
        super(TradingEnv, self).__init__()
        self.seed()
        self.current_step = window_size
        self.initial_balance = 10000
        self.df = df
        self.reward_range = (-np.inf, np.inf)
        self.window_size = window_size
        self.prices, self.volume, self.high, self.low, self.features = self._process_data()
        self.n_features = self.features.shape[1]
        self.balance = self.initial_balance
        self.position = None
        self.max_loss = max_loss
        self.total_loss = 0
        self.last_action = 1
        self.buy_and_hold = 1
        self.hold_counts = 0

        # Observation: [market_feature] + [owned_shares] + [balance]
        feature_num = self.window_size * self.n_features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_num + 1 + 1 + 1,))

        # Actions: sell, hold, buy
        self.action_space = spaces.Discrete(3)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.balance = self.initial_balance
        self.position = None
        self.last_action = 1
        self.buy_and_hold = 1
        self.hold_counts = 0
        self.total_loss = 0
        self.current_step = self.window_size
        self.initial_price = float(self.df.loc[self.current_step, 'Close'])
        return self._get_observation()

    def step(self, action):
        current_price = float(self.df.loc[self.current_step, 'Close'])
        self.date = self.df.loc[self.current_step, 'Date']
        self.buy_and_hold = current_price / self.initial_price

        if action == 0:  # hold
            reward = 0
            self.hold_counts += 1

        elif action == 1:  # buy
            if self.position is not None:
                reward = -1
            else:
                self.position = current_price
                reward = 0
                self.hold_counts = 0
                self.last_action = 1
        
        elif action == 2:  # sell
            if self.position is None:
                reward = -1
            else:
                reward = (current_price - self.position) / self.position
                self.balance += reward * self.balance
                reward =  (self.balance / self.initial_balance) - self.buy_and_hold
                self.last_action = 2
                self.hold_counts = 0
                self.position = None
        
        self.current_step += 1
        done = False
        if (self.last_action == 2 and self.balance - self.buy_and_hold * self.initial_balance < -self.max_loss):
            done = True
            reward =  (self.balance / self.initial_balance) - self.buy_and_hold
        elif self.current_step == len(self.prices):
            done = True
            reward = (self.balance / self.initial_balance) - self.buy_and_hold
        elif self.hold_counts > self.window_size:
            done = True
            reward = (self.balance / self.initial_balance) - self.buy_and_hold

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            profit = self.balance - 10000
            print(f'Step: {self.current_step}, Profit: {profit}')
        else:
            super(TradingEnv, self).render(mode=mode)


    def _get_observation(self):
        window_prices = self.prices[self.current_step - self.window_size: self.current_step]
        window_volume = self.volume[self.current_step - self.window_size: self.current_step]
        window_high = self.high[self.current_step - self.window_size: self.current_step]
        window_low = self.low[self.current_step - self.window_size: self.current_step]

        window_prices_pct_change = (window_prices - window_prices[0]) / (window_prices[0] + np.finfo(np.float32).eps)
        window_volume_pct_change = (window_volume - window_volume[0]) / (window_volume[0] + np.finfo(np.float32).eps)
        window_high_pct_change = (window_high - window_high[0]) / (window_high[0] + np.finfo(np.float32).eps)
        window_low_pct_change = (window_low - window_low[0]) / (window_low[0] + np.finfo(np.float32).eps)
      
        balance_norm = self.balance / self.initial_balance
        position = int(self.position is not None)

        data = np.concatenate([window_prices_pct_change.flatten(), window_volume_pct_change.flatten(), window_high_pct_change.flatten(), 
        window_low_pct_change.flatten(), np.array([position, balance_norm, self.last_action])])

        return data

    def _process_data(self):
        prices = self.df['Close'].values.reshape(-1, 1)
        volume = self.df['Volume'].values.reshape(-1, 1)
        high = self.df['High'].values.reshape(-1, 1)
        low = self.df['Low'].values.reshape(-1, 1)

        features = np.concatenate([np.log1p(prices), np.log1p(volume), np.log1p(high), 
                                   np.log1p(low)], axis=1)

        return prices, volume, high, low, features