import numpy as np

def logistic_loss(weights:np.ndarray, input_data:np.ndarray, bias:float)->bool:
    """
    This function calculate the Logistic regression loss.

    :param weights: the weights of the model
    :type weights: np.ndarray

    :param input_data: the input data
    :type input_data: np.ndarray

    :param bias: the bias of the model
    :type bias: float

    :return: the predicted value
    :rtype: bool

    :raises ValueError: if the length of weights does not match the number of features in input_data
    """
    if len(weights) != input_data.shape[1]:
        raise ValueError(f"weights length {len(weights)} does not match input data shape {input_data.shape[1]}")    

    linear_model = np.dot(input_data, weights) + bias
    y_predicted = 1 / (1 + np.exp(-linear_model))
    return True if y_predicted >= 0.5 else False