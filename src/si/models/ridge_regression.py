import numpy as np

from si.data.dataset import Dataset
from si.base.model import Model
from si.metrics.mse import mse


class RidgeRegression(Model):

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegression':
        X = dataset.X
        y = dataset.y

        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)

        X_centered = X - X_mean
        y_centered = y - y_mean

        identity = np.eye(X.shape[1])

        self.coef_ = np.linalg.solve(
            X_centered.T @ X_centered + self.alpha * identity,
            X_centered.T @ y_centered
        )

        self.intercept_ = y_mean - X_mean @ self.coef_

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        return dataset.X @ self.coef_ + self.intercept_

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        return mse(dataset.y, predictions)