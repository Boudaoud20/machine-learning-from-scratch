import numpy as np

def mse(y_true:np.ndarray, y_pred:np.ndarray)->float:
    """
    This function calculate the mean squared error between the true and predicted values.

    :param y_true: the true values
    :type y_true: np.ndarray

    :param y_pred: the predicted values
    :type y_pred: np.ndarray

    :return: the mean squared error
    :rtype: float
    """
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true:np.ndarray, y_pred:np.ndarray)->float:
    """
    This function calculate the root mean squared error between the true and predicted values.

    :param y_true: the true values
    :type y_true: np.ndarray

    :param y_pred: the predicted values
    :type y_pred: np.ndarray

    :return: the root mean squared error
    :rtype: float
    """
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true:np.ndarray, y_pred:np.ndarray)->float:
    """
    This function calculate the mean absolute error between the true and predicted values.

    :param y_true: the true values
    :type y_true: np.ndarray

    :param y_pred: the predicted values
    :type y_pred: np.ndarray

    :return: the mean absolute error
    :rtype: float
    """
    return np.mean(np.abs(y_true - y_pred))