import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula a precisão (accuracy) do modelo.
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total