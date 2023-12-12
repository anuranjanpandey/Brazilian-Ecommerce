import  logging
import pandas as pd
from zenml import step

@step
def evaluate_model(df: pd.DataFrame) -> None:
    """
    Evaluate the model.
    Args:
        df (pd.DataFrame): Dataframe containing the data.
    """
    logging.info("Evaluating model...")
