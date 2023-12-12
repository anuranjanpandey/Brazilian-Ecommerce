import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin, 
    X_test: pd.DataFrame, 
    y_test: pd.DataFrame
    ) -> Tuple[
        Annotated[float, "r2_score"], 
        Annotated[float, "rmse"]
    ]:
    """
    Evaluate the model.
    Args:
        model (RegressorMixin): Trained model
        X_test (pd.DataFrame): Testing data
        y_test (pd.DataFrame): Testing labels
    Returns:
        r2 (float): R2 score
        rmse (float): RMSE score    
    """
    try:
        prediction = model.predict(X_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)

        # Using the R2Score class for R2 score calculation
        r2_class = R2()
        r2_score = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)

        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)
        
        return r2_score, rmse
    except Exception as e:
        logging.error("Model evaluation failed!: {}".format(e))
        raise e
    
