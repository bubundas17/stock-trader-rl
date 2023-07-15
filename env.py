import gym
import pandas as pd
import numpy as np
from enum import Enum
from gym import spaces
from stable_baselines3 import PPO
from gym.utils import seeding

np.random.seed(0)

INITIAL_BALANCE = 10000
WINDOW_SIZE = 10
MAX_LOSS = 5000
TRAING_MODE = True

# Define an Enum class for actions


class Actions(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, window_size: int = WINDOW_SIZE, training_mode=TRAING_MODE, max_loss: int = MAX_LOSS) -> None:
        super(TradingEnv, self).__init__()
        self.seed()
        self.window_size = window_size
        self.max_loss = max_loss
        self.df = df

        # Process data
        self.prices, self.volume, self.high, self.low, self.features = self._process_data()
        self.n_features = self.features.shape[1]
        self.date = ""

        # Training mode
        self.training_mode = training_mode

        # Initialize state variables
        self._reset_state()

        # Define action and observation space
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size * self.n_features + 4,))  # all features + balance + position + buy_and_hold + last_action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._reset_state()
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        if action == Actions.HOLD.value:
            reward = 0
            self.hold_counts += 1
        elif action == Actions.BUY.value:
            reward = self._perform_buy_action()
        elif action == Actions.SELL.value:
            reward = self._perform_sell_action()

        done = self._check_done_condition()
        self._update_state()
        if(done): 
            reward = self._get_reward()
        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            profit = self.balance - INITIAL_BALANCE
            print(f'Step: {self.current_step}, Profit: {profit}')
        else:
            super(TradingEnv, self).render(mode=mode)

    def _update_state(self):
        self.date = self.df.loc[self.current_step, 'Date']
        self.last_price = self._current_price()
        self.buy_and_hold = self.last_price / self.initial_price

    def _reset_state(self):
        # Training mode
        if self.training_mode:
            self.current_step = np.random.randint(
                self.window_size, self.df.shape[0] - 1)
            self.max_steps = 366 * 2
        else:
            self.current_step = self.window_size
            self.max_steps = 0

        self.initial_step = self.current_step
        self.balance = INITIAL_BALANCE
        self.position = None
        self.hold_counts = 0
        self.buy_and_hold = 1
        self.last_action = Actions.HOLD.value
        self.initial_price = float(self.df.loc[self.current_step, 'Close'])
        self.last_price = self.initial_price

    def _current_price(self):
        return float(self.df.loc[self.current_step, 'Close'])

    def _process_data(self):
        prices = self.df['Close'].values.reshape(-1, 1)
        volume = self.df['Volume'].values.reshape(-1, 1)
        high = self.df['High'].values.reshape(-1, 1)
        low = self.df['Low'].values.reshape(-1, 1)

        features = np.concatenate([np.log1p(prices), np.log1p(
            volume), np.log1p(high), np.log1p(low)], axis=1)

        return prices, volume, high, low, features

    def _get_observation(self):
        window_data = {
            'prices': self.prices,
            'volume': self.volume,
            'high': self.high,
            'low': self.low
        }
        obs = np.array([])

        # Normalizing window data and appending to observation
        for key, values in window_data.items():
            window_values_pct_change = self._get_pct_change(values)
            obs = np.append(obs, window_values_pct_change)

        # Appending additional data to observation
        posision_pct_change = float(
            self.position / self._current_price() if self.position is not None else 0)
        balance_pct_change = self.balance / INITIAL_BALANCE
        last_action = self.last_action
        buy_and_hold = self.buy_and_hold
        added_features = [posision_pct_change,
                          balance_pct_change, last_action, buy_and_hold]

        obs = np.append(obs, added_features)

        return obs

    def _perform_buy_action(self):
        current_price = float(self.df.loc[self.current_step, 'Close'])

        if self.position is not None:
            return -1
        else:
            self.position = current_price
            self.last_action = Actions.BUY.value
            self.hold_counts = 0
            return 0.01

    def _perform_sell_action(self):
        current_price = float(self.df.loc[self.current_step, 'Close'])

        if self.position is None:
            return -1
        else:
            pct_profit = (current_price - self.position) / self.position
            self.balance += pct_profit * self.balance
            # reward = (self.balance / INITIAL_BALANCE) - self.buy_and_hold
            # if(reward > 0):
            reward = ((self.position / self._current_price()) -1) * 10
            self.last_action = Actions.SELL.value
            self.hold_counts = 0
            self.position = None
            # Add 0.1 to reward if profit is made
            return reward + 0.3 if reward > 0 else reward

    def _check_done_condition(self):
        if self.current_step >= len(self.prices) - 1:
            return True
        elif (self.last_action == Actions.SELL.value and self.balance - self.buy_and_hold * INITIAL_BALANCE < -self.max_loss):
            return True
        elif self.hold_counts > self.window_size:
            return True
        elif self.training_mode and self.max_steps != 0 and self.current_step - self.initial_step > self.max_steps:
            return True
        else:
            return False

    # Stats Logs
    def current_buy_and_hold_balance(self):
        return self.buy_and_hold * INITIAL_BALANCE

    def current_balance(self):
        return self.balance
    
    def _get_reward(self):
        return (self.balance / INITIAL_BALANCE) - self.buy_and_hold

    def _get_pct_change(self, window_values):
        """
        Helper function to calculate percentage change of window data.
        """
        values_0 = window_values[self.current_step -
                                 self.window_size: self.current_step][0]
        values_1 = window_values[self.current_step -
                                 self.window_size: self.current_step]
        return (values_1 - values_0) / (values_0 + np.finfo(np.float32).eps)
