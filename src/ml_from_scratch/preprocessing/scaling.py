import numpy as np
import pandas as pd

def scale(dataset:pd.DataFrame, columns:list[int])->None:
    """
    This function scale the dataset.

    :param dataset: the dataset to be scaled
    :type dataset: pd.DataFrame

    :param columns: the columns to be scaled
    :type columns: list[int]

    :return: none
    :rtype: None

    :raises ValueError: if the columns list is empty or longer than the number of columns in the dataset
    :raises ValueError: if the column index is out of range
    :raises ValueError: if the column index is not numeric
    """
    if len(columns) == 0:
        raise ValueError("columns list is empty")
    if len(columns) > len(dataset.columns):
        raise ValueError("columns list is longer than the number of columns in the dataset")
    
    for column in columns:
        if column < 0 or column >= len(dataset.columns):
            raise ValueError(f"column {column} is out of range")
        
        series = dataset.iloc[:, column]

        if series.dtype not in [np.float64, np.float32, np.int64, np.int32]:
            raise ValueError(f"column index {column} is not numeric")
        
        mean = series.mean()
        std = series.std()
        dataset.iloc[:, column] = (series - mean) / std if std != 0 else series - mean
    return None