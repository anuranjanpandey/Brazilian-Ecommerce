import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame,
                X_test: pd.DataFrame,
                y_train: pd.DataFrame,
                y_test: pd.DataFrame,
                config: ModelNameConfig) -> RegressorMixin:
    """
    Train the model.
    Args:
        X_train (pd.DataFrame): Training data
        X_test (pd.DataFrame): Testing data
        y_train (pd.DataFrame): Training labels
        y_test (pd.DataFrame): Testing labels
        config (ModelNameConfig): Model configuration
    """
    try:
        model = None
        if config.model_name == "LinearRegressionModel":
            mlflow.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            logging.error("Model name not found!")
            raise ValueError(
                "Model {} not supported!".format(config.model_name))
    except Exception as e:
        logging.error("Model training failed!: {}".format(e))
        raise e
