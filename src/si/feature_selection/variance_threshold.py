import numpy as np
from si.data.dataset import Dataset
from si.base.transformer import Transformer

class VarianceThreshold(Transformer):
    """
    Seleciona as características (features) cuja variância é superior a um determinado threshold
    """
    def __init__(self, threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.variance = None

    def _fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        variância de cada feature
        """
        # Calcula a variância para cada coluna de X (axis=0)
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Seleciona todas as características com variância superior ao threshold 
        e retorna um novo Dataset 
        """
        # Cria uma máscara booleana (True se a variância for estritamente maior que o threshold)
        mask = self.variance > self.threshold
        
        # Filtra o X e os nomes das features usando a máscara
        X_transformed = dataset.X[:, mask]
        features_transformed = np.array(dataset.features)[mask].tolist() if dataset.features is not None else None
        
        # Retorna um novo Dataset com os dados 
        return Dataset(X=X_transformed, y=dataset.y, features=features_transformed, label=dataset.label)