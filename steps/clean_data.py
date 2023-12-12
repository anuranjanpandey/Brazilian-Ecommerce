import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Clean the data and divide it into train and test sets.
    Args:
        df (pd.DataFrame): Dataframe containing the data.
    Returns:
        X_train (pd.DataFrame): Dataframe containing the training data.
        X_test (pd.DataFrame): Dataframe containing the test data.
        y_train (pd.Series): Series containing the training labels.
        y_test (pd.Series): Series containing the test labels.
    """
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning complete.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e