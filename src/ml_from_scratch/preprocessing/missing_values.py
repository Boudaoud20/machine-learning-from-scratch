import numpy as np
import pandas as pd

def drop_missing(dataset:pd.DataFrame)->pd.DataFrame | None:
    """
    This function drop the missing values from the dataset.

    :param dataset: the dataset to be processed
    :type dataset: pd.DataFrame

    :return: the dataset without missing values
    :rtype: pd.DataFrame
    """
    return dataset.dropna().reset_index(drop=True)

def missing_summary(dataset:pd.DataFrame)->pd.Series:
    """
    This function return a summary of the missing values in the dataset.

    :param dataset: the dataset to be processed
    :type dataset: pd.DataFrame

    :return: a summary of the missing values in the dataset
    :rtype: pd.Series
    """
    print("Missing values summary:")
    series = dataset.isnull().sum()

    return series

def fill_missing(dataset:pd.DataFrame, columns:list[int], values:list )->None:
    """
    This function fill the missing values in the dataset.

    :param dataset: the dataset to be processed
    :type dataset: pd.DataFrame

    :param columns: the columns to be processed
    :type columns: list[int]

    :param values: the values to be used to fill the missing values
    :type values: list

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
        
        series.fillna(values[column], inplace=True)  
        dataset.iloc[:, column] = series
    
    return None