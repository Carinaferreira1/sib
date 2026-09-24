import numpy as np
from typing import Callable
from si.data.dataset import Dataset
from si.base.model import Model
from si.metrics.accuracy import accuracy

# Função auxiliar para calcular a distância (caso não seja fornecida nenhuma)
def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calcula a distância euclidiana de uma amostra x e várias amostras y"""
    return np.sqrt(((x - y) ** 2).sum(axis=1))

class KNNClassifier(Model):
    """
    O algoritmo dá a classe de uma amostra com base nos k exemplos mais parecidos.
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        # Parâmetros definidos pelo utilizador
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        
        # Parâmetros definidps a partir dos dados
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        Guarda dataset de treino
        """
        self.dataset = dataset
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Estima classe para dataset de teste
        """
        predictions = []
        for x in dataset.X:
            # 1. distância entre a amostra e o dataset de treino
            distances = self.distance(x, self.dataset.X)
            
            # 2. índices dos k exemplos mais semelhantes (menor distância)
            k_nearest_idx = np.argsort(distances)[:self.k]
            
            # 3. Usa índices para recuperar as classes y do treino
            k_nearest_classes = self.dataset.y[k_nearest_idx]
            
            # 4. classe mais comum dos k exemplos
            unique, counts = np.unique(k_nearest_classes, return_counts=True)
            most_common = unique[np.argmax(counts)]
            
            predictions.append(most_common)
            
        # 5. array de previsões
        return np.array(predictions)

    def _score(self, dataset: Dataset) -> float:
        """
        accuracy
        """
        # 1. previsões
        predictions = self.predict(dataset)
        
        # 2. precisão entre os valores reais e as previsões
        return accuracy(dataset.y, predictions)