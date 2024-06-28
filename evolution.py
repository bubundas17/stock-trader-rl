import numpy as np
from deap import base, creator, tools, algorithms
from trading_env import TradingEnv

def evaluate(individual, df):
    weights1 = np.array(individual[:(5*60+3)*256]).reshape(5*60+3, 256)
    weights2 = np.array(individual[(5*60+3)*256:(5*60+3)*256 + 256*128]).reshape(256, 128)
    weights3 = np.array(individual[(5*60+3)*256 + 256*128:]).reshape(128, 3)
    
    env = TradingEnv(df)
    done = False
    
    obs = env.reset()
    
    while not done:
        hidden1 = np.tanh(np.dot(obs, weights1))
        hidden2 = np.tanh(np.dot(hidden1, weights2))
        output = np.tanh(np.dot(hidden2, weights3))
        action = np.argmax(output) - 1  # -1: Sell, 0: Hold, 1: Buy
        obs, reward, done = env.step(action)
    
    total_reward = reward  # Now we get the total reward as percentage change
    print(f"Evaluation complete. Steps: {env.current_step}, Total reward: {total_reward:.2f}%, Buy: {env.buy_count}, Sell: {env.sell_count}, Hold: {env.hold_count}")
    return (total_reward,)

def setup_evolution():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.randn)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=(5*60+3)*256 + 256*128 + 128*3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    return toolbox

def run_evolution(toolbox, df, ngen=5, population_size=50):
    population = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Record the best individual of each generation
    hof = tools.HallOfFame(1, similar=np.array_equal)

    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
        fits = toolbox.map(lambda ind: toolbox.evaluate(ind, df), offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        hof.update(population)
        record = stats.compile(population)
        
        print(f"Generation {gen+1}/{ngen}")
        print(f"  Min: {record['min']:.2f}%")
        print(f"  Max: {record['max']:.2f}%")
        print(f"  Avg: {record['avg']:.2f}%")
        print(f"  Best individual fitness: {hof[0].fitness.values[0]:.2f}%")
        print()

    best_individual = hof[0]
    return best_individual
