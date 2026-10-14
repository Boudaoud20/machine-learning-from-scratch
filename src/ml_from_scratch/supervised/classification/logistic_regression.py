import numpy as np
import pandas as pd
from numpy.typing import NDArray


class LogisticRegression :

    def __init__(self,learning_rate:float = 0.04,iteration_count:int=1000) ->None:
            self.learning_rate:float = learning_rate
            self.iteration_count: int= iteration_count
            self.weights: np.ndarray  = np.array([])
            self.bias: float | None = None

    @property
    def learning_rate(self) -> float:
        return self._learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        if not isinstance(value, (float, int)):
            raise TypeError("Learning rate must be a float or int.")
        if value <= 0:
            raise ValueError("Learning rate must be positive.")
        self._learning_rate = value

    @property
    def iteration_count(self) -> int:
        return self._iteration_count

    @iteration_count.setter
    def iteration_count(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("Iteration count must be an integer.")
        
        if value <= 0:
            raise ValueError("Iteration count must be a positive integer.")
        self._iteration_count = value



    def fit(self, x: pd.DataFrame, y_true: pd.Series) -> None:
        """
        Train the model on dataset x

        :param x: Features.
        :type x: np.DataFrame

        :param y_true: The exact values that we will predict it.
        :type y_true: pd.series

        :return: none
        :rtype: None

        :raises ValueError: Case of features are not 2D.
        :raises ValueError: Different number of samples between x and y_true.
        :raises ValueError: If the dataset is empty.

        """
        if x.ndim != 2:
            raise ValueError(
            "Input must be a two-dimensional array."
        )
        x_arr: np.ndarray= x.to_numpy(dtype=float)
        y_arr: np.ndarray= y_true.to_numpy(dtype=float)

        if x_arr.shape[0] != y_true.shape[0]:
            raise ValueError("X and y_true must contain the same number of samples.")

        if x_arr.shape[0] == 0:
            raise ValueError("Training data must not be empty.")
        
        features_count:int = x_arr.shape[1]
        rnd = np.random.default_rng() 
        
        weights: list[float] = [rnd.random() for _ in range(features_count)]  
        bias:float = rnd.random()
        self.weights = np.array(weights)
        self.bias = bias  
        for _ in range(self.iteration_count):
            y = np.asarray(
                self._prediction(weights=self.weights, input_data=x_arr, bias=self.bias),
                dtype=float,
            )
            self._compute_gradients(x=x_arr,y_pred=y, y_true= y_arr)

    def _compute_gradients(self, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        This method performs a single step of gradient descent to update the model's weights based on the predicted and true values.

        :param y_pred: A numpy array of predicted target values.
        :type y_pred: np.ndarray

        :param y_true: A numpy array of true target values.
        :type y_true: np.ndarray

        :raises ValueError: If the shapes of y_pred and y_true do not match.

        :returns: none
        :rtype: None
        """
        if y_pred.shape != y_true.shape:
            raise ValueError("Predicted and true values must have the same shape.")
        
        if len(self.weights) ==0 or self.bias is None:
            raise ValueError("Model parameters have not been initialized.")
        
        row_count: int = y_true.shape[0]

        errors: np.ndarray = y_pred - y_true

        gradient_weights = (1 / row_count) * (x.T @ errors)
        gradient_bias = (1 / row_count) * np.sum(errors)

        self.weights -= self.learning_rate * gradient_weights
        self.bias -= self.learning_rate * gradient_bias


    def _prediction(self,weights:np.ndarray, input_data:np.ndarray, bias:float)->np.ndarray| float:

        """
        This function calculate the Logistic regression loss.

        :param weights: the weights of the model
        :type weights: np.ndarray

        :param input_data: the input data
        :type input_data: np.ndarray

        :param bias: the bias of the model
        :type bias: float

        :return: the predicted values
        :rtype: np.ndarray | float

        :raises ValueError: if the inputs are not 1D or 2D array
        :raises ValueError: if the length of weights does not match the number of features in input_data
        """
        if input_data.ndim not in [1,2]:
            raise ValueError("input_data must be a 1D or 2D array")   
        feature_n:int = input_data.shape[0] if  input_data.ndim == 1 else input_data.shape[1]

        if feature_n != len(weights):
            raise ValueError("the length of weights does not match the number of features in input_data")
        
        if input_data.ndim == 1:
            linear_value: float = np.dot(input_data, weights) + bias
            predicted_value: float = 1 / (1 + np.exp(-linear_value))
            return predicted_value
        else:
            linear_values: list[float] = [np.dot(row, weights) + bias for row in input_data]
            predicted_values: list[float] = []
            for z in linear_values:
                predicted_values.append(1 / (1 + np.exp(-z)))
            return np.array(predicted_values, dtype=np.float64)



    def predict(self, x: np.ndarray, return_score: bool =False) -> np.ndarray | float | int:
        """
        This method predicts the target values for the given input features using the learned weights and bias.

        :param X: A numpy array of shape (n_samples, n_features) representing the input features.
        :return: A numpy array, containing the predicted target values.

        :raises ValueError: If the model has not been fitted yet or if the input features do not match the number of weights.
        :raises Tyy.shapepeError: If the input is not a numpy array.

        :returns: A numpy array of predicted target values.
        :rtype: np.ndarray| int
        """

        if self.weights.size == 0 or self.bias == None:
            raise ValueError("Model has not been fitted yet.")
        features_n = 1 if len(x.shape) == 1 else x.shape[1]
        if features_n != len(self.weights):
            raise ValueError("Input features do not match the number of weights.")
        
        probabilities = self._prediction(weights=self.weights, input_data=x, bias=self.bias)
        if return_score:
            return probabilities
        if isinstance(probabilities, np.ndarray):
            return (probabilities >= 0.5).astype(int)
        return int(probabilities >= 0.5)
        
