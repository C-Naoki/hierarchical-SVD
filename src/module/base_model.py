from abc import ABCMeta, abstractmethod
from typing import Any, Tuple

import numpy as np


class BaseModel(metaclass=ABCMeta):
    """
    Abstract base class for creating machine learning models.
    All machine learning　models should inherit from this class
    and implement the abstract methods.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_variables(
        self, *args: Any, **kwargs: Any  # noqa: U100  # noqa: U100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create and return the explanatory and objective variables.

        Parameters
        ----------
        *args: Any
            Variable length argument list.
        **kwargs: Any
            Arbitrary keyword arguments.

        Returns
        -------
        X, y : (Tuple[np.ndarray, np.ndarray])
            The explanatory and objective variables.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> None:  # noqa: U100
        """
        Fit the model to the data. The specific fitting procedure should be
        implemented in the classes that inherit from this class.

        Parameters
        ----------
        *args: Any
            Variable length argument list.
        **kwargs: Any
            Arbitrary keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:  # noqa: U100
        """
        Predict the output given the input data.

        Parameters
        ----------
        X_test : (np.ndarray)
            The input data for prediction.

        Returns
        -------
        y_pred : (np.ndarray)
            The predicted output.
        """
        raise NotImplementedError

    @abstractmethod
    def score(
        self, X: np.ndarray, y_true: np.ndarray  # noqa: U100  # noqa: U100
    ) -> float:
        """
        Calculate the score of the model.

        Parameters
        ----------
        X : (np.ndarray)
            The input data used for scoring.
        y_true : (np.ndarray)
            The true output data used for scoring.

        Returns
        -------
        score : (float)
            The calculated score.
        """
        raise NotImplementedError
