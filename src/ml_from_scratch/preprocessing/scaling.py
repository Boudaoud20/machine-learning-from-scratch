import numpy as np
import pandas as pd


class StandardScaler:

    def __init__(self, columns: list[int]) -> None:
        self.columns = columns
        self.means: dict[int, float] = {}
        self.stds: dict[int, float] = {}
        self.is_fitted = False

    def fit(self, dataset: pd.DataFrame) -> None:
        """
        Calculate the mean and standard deviation of the selected columns.

        :param dataset: The dataset used to calculate scaling parameters.
        :type dataset: pd.DataFrame

        :raises ValueError: If the columns list is empty or too long.
        :raises ValueError: If a column index is out of range or not numeric.
        """
        if len(self.columns) == 0:
            raise ValueError("columns list is empty")

        if len(self.columns) > len(dataset.columns):
            raise ValueError(
                "columns list is longer than the number of columns in the dataset"
            )

        for column in self.columns:
            if column < 0 or column >= len(dataset.columns):
                raise ValueError(f"column {column} is out of range")

            series = dataset.iloc[:, column]

            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(f"column index {column} is not numeric")

            self.means[column] = float(series.mean())
            self.stds[column] = float(series.std())

        self.is_fitted = True

    def transform(self, dataset: pd.DataFrame) -> None:
        """
        Scale the selected columns in place using previously calculated parameters.

        :param dataset: The dataset to be scaled.
        :type dataset: pd.DataFrame

        :return: None
        :rtype: None

        :raises ValueError: If the scaler has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet.")

        for column in self.columns:
            if column < 0 or column >= len(dataset.columns):
                raise ValueError(f"column {column} is out of range")

            series = dataset.iloc[:, column]
            mean = self.means[column]
            std = self.stds[column]

            dataset.iloc[:, column] = (
                (series - mean) / std if std != 0 else series - mean
            )


    def fit_transform(self, dataset: pd.DataFrame) -> None:
        """
        Calculate scaling parameters and scale the dataset in place.

        :param dataset: The dataset to be scaled.
        :type dataset: pd.DataFrame

        :return: None
        :rtype: None
        """
        self.fit(dataset)
        self.transform(dataset)