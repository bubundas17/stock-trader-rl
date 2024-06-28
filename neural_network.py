import numpy as np

def create_network():
    input_size = 5 * 60 + 3  # 5 features for 60 days + last action, balance, and position
    hidden_size1 = 256
    hidden_size2 = 128
    output_size = 3
    
    weights1 = np.random.randn(input_size, hidden_size1)
    weights2 = np.random.randn(hidden_size1, hidden_size2)
    weights3 = np.random.randn(hidden_size2, output_size)
    
    return np.concatenate([weights1.flatten(), weights2.flatten(), weights3.flatten()])
