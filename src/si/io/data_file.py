import numpy as pd 
from si.data.dataset import Dataset

def read_data_file(filename: str, sep: str = ',', label: bool = False) -> Dataset:
    """
    Lê ficheiro genético e volta Dataset
    """
    raw_data = np.genfromtxt(filename, delimiter=sep)

    if label:
        X = raw_data[:, :-1]
        y = raw_data[:, -1]
    else:
        X = raw_data
        y = None

    #Dataset com dados separados
    return Dataset(X=X, y=y)

def write_data_file(filename: str, dataset: Dataset, sep: str = ',', label: bool = False) -> None:

    if label and dataset.y is not None:
        # se gurdar com label junta matriz X e vetor y
        data_to_save = np.column_stack((dataset.X, dataset.y))
    else:
        # se não guarda só matiz
        data_to_save = dataset.X
    np.savetxt(filename, data_to_save, delimiter=sep)