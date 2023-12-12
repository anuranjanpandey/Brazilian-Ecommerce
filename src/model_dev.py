import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """Abstract class for all models"""
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.DataFrame): Training labels
        """
        pass


class LinearRegressionModel(Model):
    """Linear Regression model"""
    def train(self, X_train, y_train, **kwargs):
        """Train the model
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.DataFrame): Training labels
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model trained!")
            return reg
        except Exception as e:
            logging.error("Model training failed!: {}".format(e))
            raise e