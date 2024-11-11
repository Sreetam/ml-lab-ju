import kagglehub
import os

# Define the target download path as 'datasets' folder
target_path = os.path.join(os.path.dirname(__file__), 'datasets')

# Download the dataset to the specified folder
dataset_path = kagglehub.dataset_download("yasserh/housing-prices-dataset", path=target_path)

# Display the path to the downloaded files
print(f"Dataset downloaded to: {dataset_path}")
