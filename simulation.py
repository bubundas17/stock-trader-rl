import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# Load the trained model
model = nn.Sequential(
    nn.Linear(3, 128),
    nn.ReLU(),
    nn.Linear(128, 3),
    nn.Softmax(dim=1)
)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Load financial data
df = pd.read_csv('stock_data.csv')

# Define state variables from data
def get_state(data, t, last_action):
    window = data.iloc[t-365:t]  # 1-year window
    prev_close = window['Close'].values[:-1]
    prev_volume = window['Volume'].values[:-1]

    state = np.column_stack((prev_close, prev_volume, last_action))
    return state

# Actions: 0 = Sell, 1 = Hold, 2 = Buy
n_actions = 3

# Simulate trading using the trained model
starting_balance = 10000  # Starting balance in USD
last_action = 0  # Initial action is sell
balance = starting_balance

for t in range(366, len(df)):
    state = get_state(df, t, last_action)

    # Get action probabilities
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action_probs = model(state_tensor)
    action = torch.argmax(action_probs)

    # Get current stock price
    current_price = df.iloc[t]['Close']

    # Take action and calculate balance
    if action == 0:  # Sell
        if last_action == 2:  # Only allow selling after a buy action
            shares = balance / current_price  # Sell all shares
            balance -= current_price * shares  # Subtract the cost from the balance
            balance *= 0.99  # Apply a 1% transaction fee
            last_action = 0
    elif action == 2:  # Buy
        if last_action == 0:  # Only allow buying after a sell action
            shares = balance / current_price  # Buy maximum number of shares
            balance -= current_price * shares  # Subtract the cost from the balance
            balance *= 0.99  # Apply a 1% transaction fee
            last_action = 2
    else:  # Hold
        pass

# Print the final balance
print(f"Final balance: {balance}")
