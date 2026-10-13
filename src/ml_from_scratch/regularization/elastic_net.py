from .l1 import l1_gradient, l1_penalty
from .l2 import l2_gradient, l2_penalty


import numpy as np

def elastic_net_penalty(weights: np.ndarray,l1_lambda: float,l2_lambda: float,) -> float:
    """
    Calculate the Elastic Net regularization penalty.

    The penalty is the sum of the L1 and L2 regularization penalties.

    :param weights: The model's weights.
    :type weights: np.ndarray

    :param l1_lambda: The regularization parameter for the L1 penalty.
    :type l1_lambda: float

    :param l2_lambda: The regularization parameter for the L2 penalty.
    :type l2_lambda: float

    :return: The combined L1 and L2 regularization penalty.
    :rtype: float
    """
    return l1_penalty(weights, l1_lambda) + l2_penalty(weights, l2_lambda)


def elastic_net_gradient(weights: np.ndarray,l1_lambda: float,l2_lambda: float,) -> np.ndarray:
    """
    Calculate the Elastic Net regularization gradient.

    The gradient is the sum of the L1 and L2 regularization gradients.

    :param weights: The model's weights.
    :type weights: np.ndarray

    :param l1_lambda: The regularization parameter for the L1 gradient.
    :type l1_lambda: float

    :param l2_lambda: The regularization parameter for the L2 gradient.
    :type l2_lambda: float

    :return: The combined L1 and L2 regularization gradient.
    :rtype: np.ndarray
    """
    return l1_gradient(weights, l1_lambda)+l2_gradient(weights, l2_lambda)
    
    
