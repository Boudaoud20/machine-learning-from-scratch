import pandas as pd
import numpy as np

def split(dataset:pd.DataFrame ,train_percent:float, validation_percent:float | int = 0 , test_percent:float | int = 0 , )->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function split the dataset into tree parts (train, validation, test).

    :param train_percent: the percent of the dataset that will be used for training
    :type train_percent: float

    :param validation_percent: the percent of the dataset that will be used for validation
    :type validation_percent: float

    :param test_percent: the percent of the dataset that will be used for testing
    :type test_percent: float

    :param dataset: the dataset to be split
    :type dataset: pd.DataFrame

    :return: a tuple of three datasets (train, validation, test)
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

    :raises ValueError: if the sum of train_percent, validation_percent and test_percent is not equal to 1
    :raises ValueError: if any of the percents is not between 0 and 1
    """
    if train_percent > 1 or train_percent < 0 :
        raise ValueError(f"{train_percent} is a wrong percent")
    
    if  validation_percent > 1 or validation_percent < 0:
        raise ValueError(f"{validation_percent} is a wrong percent")
    
    if test_percent > 1 or test_percent < 0:
        raise ValueError(f"{test_percent} is a wrong percent")
    
    rows = len(dataset)
    train_count = int(rows * train_percent)
    validation_count = int(rows * validation_percent)
    test_count = int(rows * test_percent)
    
    #wrong one
    #return (dataset[0:train_count,::], dataset[train_count:train_count+validation_count,::], dataset[train_count+validation_count:train_count+validation_count+test_count,::])

    return (dataset.iloc[:train_count, :],dataset.iloc[train_count:train_count + validation_count, :],dataset.iloc[train_count + validation_count:train_count + validation_count + test_count, :])


def get_input_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate a dataset into input features and target values.

    :param dataset: The dataset to be separated.
    :type dataset: pd.DataFrame

    :return: A tuple containing the input features and target values.
    :rtype: tuple[pd.DataFrame, pd.Series]

    :raises ValueError: If the number of columns is less then 2.
    """
    column_n : int = len(dataset.columns)

    if column_n < 2:
        raise ValueError(f"number of columns is {column_n}")
    return (dataset.iloc[:, :-1],dataset.iloc[:, -1].squeeze())
