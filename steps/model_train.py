import  logging
import pandas as pd
from zenml import step

@step
def train_model(df: pd.DataFrame) -> None:
    """
    Train the model.
    Args:
        df (pd.DataFrame): Dataframe containing the data.
    """
    logging.info("Training model...")