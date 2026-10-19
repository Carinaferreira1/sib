import numpy as np
from si.data.dataset import Dataset


def read_data_file(filename: str, sep: str = None, label: bool = False) -> Dataset:
    """
    Lê um ficheiro e devolve um Dataset.
    """

    raw_data = np.genfromtxt(filename, delimiter=sep)

    if label:
        X = raw_data[:, :-1]
        y = raw_data[:, -1]
    else:
        X = raw_data
        y = None

    return Dataset(X=X, y=y)


def write_data_file(filename: str, dataset: Dataset, sep: str = None, label: bool = False) -> None:
    """
    Guarda os dados de um Dataset num ficheiro DATA.
    """

    if sep is None:
        sep = " "

    if label:
        data = np.column_stack((dataset.X, dataset.y))
    else:
        data = dataset.X

    np.savetxt(filename, data, delimiter=sep)