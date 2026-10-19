import numpy as np
from typing import Callable

from si.data.dataset import Dataset
from si.base.model import Model
from si.metrics.rmse import rmse
from si.models.knn_classifier import euclidean_distance


class KNNRegressor(Model):

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        self.dataset = dataset
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        predictions = []

        for x in dataset.X:
            distances = self.distance(x, self.dataset.X)
            k_nearest_idx = np.argsort(distances)[:self.k]
            k_nearest_values = self.dataset.y[k_nearest_idx]

            prediction = np.mean(k_nearest_values)
            predictions.append(prediction)

        return np.array(predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        return rmse(dataset.y, predictions)