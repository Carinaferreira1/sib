import pandas as pd
from si.data.dataset import Dataset

def read_csv(filename: str, sep: str = ',', features:bool = False, label = bool = False) -> Dataset:
    """
    Lê um ficheiro CSV e devolve um objeto Dataset
    """
    # se tiver featrures entáo a 1 linha é o cabeçalho
    header = 0 if features else None
    df = pd.read_csv(filename, sep=sep, header=header)

    if label:
        # se tiver label então: x tudo menos a ultima coluna e y é a última
        X = df.iloc[:, :-1].values
        y = df.iloc[:, :-1].values

        if features:
            feature_names = list(df.columns[:-1])
            label_name = str(df.columns[-1])
        else:
            feature_names = None
            label_name = None
    else:
        # sem label o x é tudo e nao tem y
        X = df.values
        y = None
        feature_names = list(df.columns) if features else None
        label_name = None
    # Faz e devolve dataset
    return Dataset(X=X, y=y, features=feature_names, label=label_name)

def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = False, label: bool = False) -> None:
    """
    Escreve um sataset num CSV
    """
    if hasattr(dataset, 'to_dataframe'):
        df = dataset.to_dataframe()
    else:
        import numpy as np
        if dataset.y is not None and label:
            data = np.c_[dataset.X, dataset.y]
            cols = list(dataset.features) if (features and dataset.features is not None) else None
            if cols is not None:
                cols.append(dataset.label if dataset.label else 'y')
        else:
            data = dataset.X
            cols = list(dataset.features) if (features and dataset.features is not None) else None

        df = pd.DataFrame(data, columns=cols)

    # Guardar ficheiro
    df.to_csv(filename, sep=sep, index=False, header=features)