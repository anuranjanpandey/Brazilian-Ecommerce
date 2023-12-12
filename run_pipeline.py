from pipelines.training_pipeline import training_pipeline


if __name__ == "__main__":
    # Path to the data
    data_path = 'data/olist_customers_dataset.csv'

    # Run the pipeline
    training_pipeline(data_path)