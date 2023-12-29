import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """Abstract class defining strategies for evaluation of our models"""
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate the score of the model
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        """
        pass


class MSE(Evaluation):
    """Evaluation strategy that uses Mean Squared Error"""

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate the score of the model
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        """
        try:
            logging.info("Calculating MSE...")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("MSE calculation failed!: {}".format(e))
            raise e


class R2(Evaluation):
    """Evaluation strategy that uses R2"""

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate the score of the model
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        """
        try:
            logging.info("Calculating R2...")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("R2 calculation failed!: {}".format(e))
            raise e


class RMSE(Evaluation):
    """Evaluation strategy that uses Root Mean Squared Error"""

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate the score of the model
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        """
        try:
            logging.info("Calculating RMSE...")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("RMSE calculation failed!: {}".format(e))
            raise e
