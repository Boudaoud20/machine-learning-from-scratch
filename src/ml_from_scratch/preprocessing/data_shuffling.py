import pandas as pd
import numpy as np

def shuffle(dataset:pd.DataFrame, seed:int = 0)->pd.DataFrame:
    """
    This function shuffle the dataset.

    :param dataset: the dataset to be shuffled
    :type dataset: pd.DataFrame

    :param seed: the seed for the random number generator
    :type seed: int

    :return: the shuffled dataset
    :rtype: pd.DataFrame
    """
    rnd = np.random.default_rng(seed)
    return dataset.sample(frac=1, random_state=rnd).reset_index(drop=True)