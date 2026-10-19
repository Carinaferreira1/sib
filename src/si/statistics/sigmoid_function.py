import numpy as np

def sigmoid_function(X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-X))