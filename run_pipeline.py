from pipelines.training_pipeline import training_pipeline
from zenml.client import Client


if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    # Path to the data
    data_path = 'data/olist_customers_dataset.csv'

    # Run the pipeline
    training_pipeline(data_path)

# mlflow ui --backend-store-uri "file:/home/anuranjan/.config/zenml/local_stores/cb062da0-879f-4981-8e21-d87b460a2fda/mlruns"