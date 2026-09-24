import numpy as np
from typing import Tuple
from si.data.dataset import Dataset

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Divide em dados de treino e dados de teste.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = dataset.X.shape[0]
    n_test = int(n_samples * test_size)
    permutations = np.random.permutation(n_samples)
    
    test_idxs = permutations[:n_test]
    train_idxs = permutations[n_test:]
    
    train_X = dataset.X[train_idxs]
    train_y = dataset.y[train_idxs] if dataset.y is not None else None
    train_dataset = Dataset(X=train_X, y=train_y, features=dataset.features, label=dataset.label)
    
    test_X = dataset.X[test_idxs]
    test_y = dataset.y[test_idxs] if dataset.y is not None else None
    test_dataset = Dataset(X=test_X, y=test_y, features=dataset.features, label=dataset.label)
    
    return train_dataset, test_dataset

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Divide o dataset em treino e teste estratificado
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # classes únicas e a contagem de cada uma
    unique_labels, counts = np.unique(dataset.y, return_counts=True)
    
    train_idxs = []
    test_idxs = []
    
    # Faz loop por classe para calcular e dividir
    for label, count in zip(unique_labels, counts):
        # Número de amostras de teste para a classe atual
        n_test = int(count * test_size)
        
        # índices das amostras que pertencem à classe e baralha
        label_idxs = np.where(dataset.y == label)[0]
        np.random.shuffle(label_idxs)
        
        # Adiciona os índices de teste e de treino 
        test_idxs.extend(label_idxs[:n_test])
        train_idxs.extend(label_idxs[n_test:])
        
    train_dataset = Dataset(X=dataset.X[train_idxs], y=dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test_dataset = Dataset(X=dataset.X[test_idxs], y=dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    
    return train_dataset, test_dataset