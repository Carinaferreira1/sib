import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):

    def __init__(self, l2_penalty: float = 1.0,
                 alpha: float = 0.01,
                 max_iter: int = 1000,
                 patience: int = 100,
                 scale: bool = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'LogisticRegression':
        X = dataset.X.astype(float)
        y = dataset.y.astype(float)
        m, n = X.shape

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1

        if self.scale:
            X = (X - self.mean) / self.std

        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        self.cost_history = {}

        best_cost = float("inf")
        no_improvement = 0

        for i in range(self.max_iter):
            probabilities = sigmoid_function(X @ self.theta + self.theta_zero)
            error = probabilities - y

            gradient = (X.T @ error) / m
            gradient += (self.l2_penalty / m) * self.theta
            gradient_zero = np.mean(error)

            self.theta -= self.alpha * gradient
            self.theta_zero -= self.alpha * gradient_zero

            current_cost = self.cost(X, y)
            self.cost_history[i] = current_cost

            if current_cost < best_cost - 1e-8:
                best_cost = current_cost
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= self.patience:
                break

        return self

    def cost(self, X: np.ndarray, y: np.ndarray) -> float:
        m = len(y)

        probabilities = sigmoid_function(X @ self.theta + self.theta_zero)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

        log_loss = -np.mean(
            y * np.log(probabilities) +
            (1 - y) * np.log(1 - probabilities)
        )

        regularization = (
            self.l2_penalty / (2 * m)
        ) * np.sum(self.theta ** 2)

        return float(log_loss + regularization)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        X = dataset.X.astype(float)

        if self.scale:
            X = (X - self.mean) / self.std

        probabilities = sigmoid_function(X @ self.theta + self.theta_zero)

        return (probabilities >= 0.5).astype(int)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        return accuracy(dataset.y, predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        return accuracy(dataset.y, predictions)