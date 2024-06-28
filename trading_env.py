import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class TradingEnv:
    def __init__(self, data, initial_balance=100000, max_steps=500, window_size=60):
        self.data = data
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data[['Open', 'High', 'Low', 'Close', 'Volume']])
        self.total_reward = 0
        self.enable_history = False
        self.reset()

        # Brokerage charges
        self.brokerage_flat_fee = 20  # Flat fee per trade
        self.brokerage_percent = 0.0003  # 0.03% of the transaction amount
        self.stt_percent = 0.001  # 0.1% of the transaction amount for equity delivery
        self.sebi_fees = 0.00005  # 0.005% of the transaction amount
        self.gst_percent = 0.18  # 18% GST on brokerage

    def set_max_steps(self, steps):
        self.max_steps = steps
    
    def set_history_record(self, history_enabled=False):
        self.enable_history = history_enabled
    
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = self.window_size
        self.history = []
        self.total_reward = 0
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0
        self.last_action = 0  # Initialize the last action as hold
        return self._get_observation()

    def _calculate_charges(self, transaction_amount):
        brokerage = max(self.brokerage_flat_fee, transaction_amount * self.brokerage_percent)
        stt = transaction_amount * self.stt_percent
        sebi_fee = transaction_amount * self.sebi_fees
        gst = brokerage * self.gst_percent
        total_charges = brokerage + stt + sebi_fee + gst
        return total_charges

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        next_price = self.data.iloc[self.current_step + 1]['Close']
        actual_action = 0
        if action == 1 and self.balance > 0:  # Buy
            shares = self.balance / current_price  # Buy at most 1 share
            transaction_amount = shares * current_price
            charges = self._calculate_charges(transaction_amount)
            self.position += shares
            self.balance -= (transaction_amount + charges)
            self.buy_count += 1
            actual_action = 1 
            
        elif action == -1 and self.position > 0:  # Sell
            transaction_amount = self.position * current_price
            charges = self._calculate_charges(transaction_amount)
            self.balance += (transaction_amount - charges)
            self.position = 0
            self.sell_count += 1
            actual_action = -1 
        else:  # Hold
            self.hold_count += 1
            actual_action = 0

        self.last_action = action
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1 or self.current_step >= self.max_steps
        
        if self.enable_history:
            self.history.append((self.current_step, self.balance + self.position * next_price, actual_action))

        if done:
            final_assets = self.balance + self.position * next_price
            self.total_reward = (final_assets - self.initial_balance) / self.initial_balance * 100
            # Overriding reward for buy and hold.
            if self.buy_count < 5 :
                self.total_reward = 0
            return self._get_observation(), self.total_reward, done
        else:
            return self._get_observation(), 0, done

    def _get_observation(self):
        obs = self.data.iloc[self.current_step - self.window_size:self.current_step][['Open', 'High', 'Low', 'Close', 'Volume']]
        normalized_obs = self.scaler.transform(obs)
        flattened_obs = normalized_obs.flatten()
        
        # Append last action, current balance, and position to the observation
        additional_features = np.array([self.last_action, self.balance / self.initial_balance, self.position > 1])
        return np.concatenate([flattened_obs, additional_features])
