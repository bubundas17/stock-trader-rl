import data_preprocessing
import neural_network
import evolution
import visualization

def main():
    # Load and preprocess data
    df = data_preprocessing.load_data('test_indexes.csv')
    
    # Set up evolutionary algorithm
    toolbox = evolution.setup_evolution()
    
    # Run evolution
    best_individual = evolution.run_evolution(toolbox, df, ngen=5, population_size=50)
    
    # Visualize performance
    visualization.visualize_performance(best_individual, df)
    
    # Plot evolution stats
    stats = toolbox.stats
    visualization.plot_evolution_stats(stats, ngen=5)

if __name__ == "__main__":
    main()
