import numpy as np

def l1_penalty(weights:np.ndarray, lambda_:float)->float:
    """
    calculate the L1 regression penality.

    :param weight: the weights of the equation.
    :type weight: np.ndarray

    :param lambda_: Regularization strength
    :type lambda_: float

    :return: penality value
    :rtype: float
    """
    return float(lambda_ * np.sum(np.abs(weights)))

def l1_gradient(weights:np.ndarray, lambda_:float):
    """
    calculate the L1 regularization subgradient.

    :param weight: the weights of the equation.
    :type weight: np.ndarray

    :param lambda_: Regularization strength
    :type lambda_: float

    :return: penality value
    :rtype: float
    """
    return lambda_ * np.sign(weights)