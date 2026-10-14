import numpy as np

def bce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Binary Cross-Entropy (BCE) loss.

    :param y_pred: Array of predicted probabilities between 0 and 1.
    :type y_pred: np.ndarray

    :param y_true: Array exact label values.
    :type y_true: np.ndarray

    :return: The mean binary cross-entropy loss.
    :rtype: float

    :raises ValueError: If the arrays have incompatible shapes, are empty.
    :raises ValueError: contain invalid labels, or contain invalid probabilities.
    :raises TypeError: If the inputs are not NumPy arrays.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be NumPy arrays.")

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Input arrays must not be empty.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1.")

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred must contain only finite values.")

    if np.any((y_pred < 0) | (y_pred > 1)):
        raise ValueError("y_pred must contain probabilities between 0 and 1.")

    epsilon = 1e-15
    y_pred = np.clip(y_pred.astype(float), epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return float(loss)