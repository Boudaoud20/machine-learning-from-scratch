import numpy as np

def l2_penalty(weights:np.ndarray, lambda_:float)->float:
    """
    calculate the L2 regression penality.

    :param weight: the weights of the equation.
    :type weight: np.ndarray

    :param lambda_: Regularization strength
    :type lambda_: float

    :return: penality value
    :rtype: float
    """
    return float(lambda_ * np.sum(weights**2))

def l2_gradient(weights:np.ndarray, lambda_:float):
    """
    calculate the L2 regularization subgradient.

    :param weight: the weights of the equation.
    :type weight: np.ndarray

    :param lambda_: Regularization strength
    :type lambda_: float

    :return: penality value
    :rtype: float
    """
    return 2 * lambda_ * weights