import matplotlib.pyplot as plt
import numpy as np
from trading_env import TradingEnv

def visualize_performance(best_individual, df):
    weights1 = np.array(best_individual[:(5*60+3)*256]).reshape(5*60+3, 256)
    weights2 = np.array(best_individual[(5*60+3)*256:(5*60+3)*256 + 256*128]).reshape(256, 128)
    weights3 = np.array(best_individual[(5*60+3)*256 + 256*128:]).reshape(128, 3)
    
    env = TradingEnv(df)
    done = False
    env.set_max_steps(5000)
    env.set_history_record(True)
    obs = env.reset()
    while not done:
        hidden1 = np.tanh(np.dot(obs, weights1))
        hidden2 = np.tanh(np.dot(hidden1, weights2))
        output = np.tanh(np.dot(hidden2, weights3))
        action = np.argmax(output) - 1  # -1: Sell, 0: Hold, 1: Buy
        obs, reward, done = env.step(action)
    
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

def plot_evolution_stats(stats, ngen):
    gen = range(1, ngen + 1)
    fit_mins = [stats.compile(population)['min'] for population in gen]
    fit_avgs = [stats.compile(population)['avg'] for population in gen]
    fit_maxs = [stats.compile(population)['max'] for population in gen]

    plt.figure(figsize=(12, 6))
    plt.plot(gen, fit_mins, label="Minimum")
    plt.plot(gen, fit_avgs, label="Average")
    plt.plot(gen, fit_maxs, label="Maximum")
    plt.title("Evolution of Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (% Change)")
    plt.legend()
    plt.show()
