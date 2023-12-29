import logging
import pandas as pd
from zenml import step


class IngestData:
    """
    Ingest data from the data_path.
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): Path to the data.
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingest data from the data_path.
        Returns:
            pd.DataFrame: Dataframe containing the data.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingest data from the data_path.
    Args:
        data_path (str): Path to the data.
    Returns:
        pd.DataFrame: Dataframe containing the data.
    """
    return IngestData(data_path).get_data()
