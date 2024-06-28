import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
from collections import deque

# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
print("Loading data...")
df = pd.read_csv('test_indexes.csv', parse_dates=['Date'])
print(f"Loaded {len(df)} data points")
df['Returns'] = df['Close'].pct_change()

# Define the trading environment
class TradingEnv:
    def __init__(self, data, initial_balance=100000, max_steps=4000, window_size=60):
        self.data = data
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.window_size = window_size
        self.scaler = MinMaxScaler()

        # Convert DataFrame to NumPy arrays
        self.price_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        self.close_prices = data['Close'].values
        self.observation_window = deque(maxlen=window_size)
        # Fit the scaler on the entire dataset once
        self.scaler.fit(self.price_data)
        self.enable_history = False
        self.reset()

        # Brokerage charges
        self.brokerage_flat_fee = 20
        self.brokerage_percent = 0.0003
        self.stt_percent = 0.001
        self.sebi_fees = 0.00005
        self.gst_percent = 0.18

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
        self.last_action = 0
        self.observation_window.clear()
        for i in range(self.window_size):
            self.observation_window.append(self.price_data[i])

        return self._get_observation()

    def _calculate_charges(self, transaction_amount):
        brokerage = np.maximum(self.brokerage_flat_fee, transaction_amount * self.brokerage_percent)
        stt = transaction_amount * self.stt_percent
        sebi_fee = transaction_amount * self.sebi_fees
        gst = brokerage * self.gst_percent
        return brokerage + stt + sebi_fee + gst

    def step(self, action):
        current_price = self.close_prices[self.current_step]
        next_price = self.close_prices[self.current_step + 1]

        if action == 1 and self.balance > 0:  # Buy
            shares = self.balance / current_price
            transaction_amount = shares * current_price
            charges = self._calculate_charges(transaction_amount)
            self.position += shares
            self.balance -= (transaction_amount + charges)
            self.buy_count += 1
        elif action == -1 and self.position > 0:  # Sell
            transaction_amount = self.position * current_price
            charges = self._calculate_charges(transaction_amount)
            self.balance += (transaction_amount - charges)
            self.position = 0
            self.sell_count += 1
        else:  # Hold
            self.hold_count += 1

        self.last_action = action
        self.current_step += 1
        self.observation_window.append(self.price_data[self.current_step])
        done = self.current_step >= len(self.data) - 1 or self.current_step >= self.max_steps

        if self.enable_history:
            self.history.append((self.current_step, self.balance + self.position * next_price, action))

        if done:
            final_assets = self.balance + self.position * next_price
            self.total_reward = (final_assets - self.initial_balance) / self.initial_balance * 100
            if self.buy_count < 5:
                self.total_reward = 0
            
            # self.total_reward = self.total_reward * ((self.buy_count / (self.window_size / 2 ) * 0.3) +1)

            return self._get_observation(), self.total_reward  , True
        else:
            return self._get_observation(), 0, done

    def _get_observation(self):
        normalized_obs = self.scaler.transform(np.array(self.observation_window))
        flattened_obs = normalized_obs.flatten()
        additional_features = np.array([self.last_action, self.balance / self.initial_balance, self.position > 1])
        return np.concatenate([flattened_obs, additional_features])

# Define the neural network using PyTorch
class TradingNetwork(nn.Module):
    def __init__(self):
        super(TradingNetwork, self).__init__()
        self.fc1 = nn.Linear(5 * 60 + 3,512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# Evaluation function
def evaluate(individual):
    env = TradingEnv(df)
    done = False
    model = TradingNetwork().to(device)
    obs = env.reset()

    # Update model with individual's weights
    state_dict = model.state_dict()
    weight_keys = list(state_dict.keys())
    start = 0

    for key in weight_keys:
        shape = state_dict[key].shape
        size = np.prod(shape)
        state_dict[key] = torch.tensor(individual[start:start + size].reshape(shape)).float().to(device)
        start += size

    model.load_state_dict(state_dict)
    model.eval()

    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    while not done:
        with torch.no_grad():
            output = model(obs)
        action = torch.argmax(output).item() - 1
        obs, reward, done = env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    total_reward = reward
    print(f"Evaluation complete. Steps: {env.current_step}, Total reward: {total_reward:.2f}%, Buy: {env.buy_count}, Sell: {env.sell_count}, Hold: {env.hold_count}")
    return (total_reward,)

# Custom comparison function for individuals
def compare_individuals(ind1, ind2):
    return np.array_equal(ind1, ind2)

# Set up the evolutionary algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.randn)
individual_size = sum(p.numel() for p in TradingNetwork().parameters())
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=individual_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Save checkpoint
def save_checkpoint(population, generation, hof, timestamp):
    checkpoint = {
        'population': population,
        'generation': generation,
        'hof': hof,
    }
    with open(f'models/checkpoint_{timestamp}.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved at generation {generation}")

# Load checkpoint
def load_checkpoint(filepath):
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint['population'], checkpoint['generation'], checkpoint['hof']

# Visualization function
def visualize_performance(best_individual, df):
    model = TradingNetwork().to(device)
    state_dict = model.state_dict()
    weight_keys = list(state_dict.keys())
    start = 0

    for key in weight_keys:
        shape = state_dict[key].shape
        size = np.prod(shape)
        state_dict[key] = torch.tensor(best_individual[start:start + size].reshape(shape)).float().to(device)
        start += size

    model.load_state_dict(state_dict)
    model.eval()

    env = TradingEnv(df)
    done = False
    env.set_max_steps(5000)
    env.set_history_record(True)
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    while not done:
        with torch.no_grad():
            output = model(obs)
        action = torch.argmax(output).item() - 1
        obs, reward, done = env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    history = np.array(env.history)

    plt.figure(figsize=(12, 6))
    plt.plot(history[:, 0], history[:, 1], label='Portfolio Value')
    plt.plot(df.index[60:], df['Close'][60:] * env.initial_balance / df['Close'][60], label='Buy and Hold Strategy')

    buy_points = history[history[:, 2] == 1]
    sell_points = history[history[:, 2] == -1]

    plt.scatter(buy_points[:, 0], buy_points[:, 1], color='g', marker='^', label='Buy')
    plt.scatter(sell_points[:, 0], sell_points[:, 1], color='r', marker='v', label='Sell')

    plt.title('Agent Performance vs Buy and Hold Strategy')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

    print(f"Final buy count: {env.buy_count}")
    print(f"Final sell count: {env.sell_count}")
    print(f"Final hold count: {env.hold_count}")

# Create directories if not exist
os.makedirs('models', exist_ok=True)

# Run the evolution
print("Starting evolution...")
ngen = 300 
population_file = 'checkpoint.pkl'

if os.path.exists(population_file):
    population, start_gen, hof = load_checkpoint(population_file)
    print(f"Resuming from generation {start_gen}")
else:
    population = toolbox.population(n=100)
    start_gen = 0
    hof = tools.HallOfFame(1, similar=compare_individuals)

# Reinitialize statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

for gen in range(start_gen, ngen):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    hof.update(population)
    record = stats.compile(population)

    print(f"Generation {gen + 1}/{ngen}")
    print(f"  Min: {record['min']:.2f}%")
    print(f"  Max: {record['max']:.2f}%")
    print(f"  Avg: {record['avg']:.2f}%")
    print(f"  Best individual fitness: {hof[0].fitness.values[0]:.2f}%")
    print()

    # Save checkpoint after each generation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if gen%5 == 0:
        save_checkpoint(population, gen + 1, hof, timestamp)

best_individual = hof[0]
print(f"Best individual fitness: {best_individual.fitness.values[0]:.2f}%")

# Visualize the performance
print("Visualizing performance...")
visualize_performance(best_individual, df)

# Plot the evolution of fitness
gen = range(1, ngen + 1)
fit_mins = [stats.compile(tools.selBest(population, len(population)))['min'][0] for gen in range(start_gen, ngen)]
fit_avgs = [stats.compile(tools.selBest(population, len(population)))['avg'][0] for gen in range(start_gen, ngen)]
fit_maxs = [stats.compile(tools.selBest(population, len(population)))['max'][0] for gen in range(start_gen, ngen)]

plt.figure(figsize=(12, 6))
plt.plot(gen, fit_mins, label="Minimum")
plt.plot(gen, fit_avgs, label="Average")
plt.plot(gen, fit_maxs, label="Maximum")
plt.title("Evolution of Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness (% Change)")
plt.legend()
plt.show()
