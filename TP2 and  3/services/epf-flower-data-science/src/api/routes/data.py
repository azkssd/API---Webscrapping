from fastapi import APIRouter
import os
import opendatasets as od
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.services.data import load_iris_dataset2, process_iris_dataset, split_iris_dataset, train_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

router = APIRouter()

# Define the directory to save the dataset
DATA_DIR = "src/data"
# Define the path of the saved Iris.csv
DATA_FILE = os.path.join("src/data/iris/Iris.csv")

# Create a route to download the dataset
@router.get("/download-iris")
async def download_iris_dataset():
    """
    Downloads the Iris dataset from Kaggle and saves it to the src/data directory.
    """
    dataset_url = "https://www.kaggle.com/datasets/uciml/iris"
    
    try:
        # Ensure the data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        # Download the dataset
        od.download(dataset_url, DATA_DIR)

        return {"message": "Dataset downloaded successfully!", "path": DATA_DIR}
    except Exception as e:
        return {"error": str(e)}

# Create a route to load the dataset as a DataFrame and return it as JSON
@router.get("/load-iris")
async def load_iris_dataset():
    """
    Loads the Iris dataset file as a pandas DataFrame and returns it as JSON.
    """
    try:
        # Check if the dataset file exists
        if not os.path.exists(DATA_FILE):
            return {"error": f"The dataset file was not found at {DATA_FILE}. Please download it first."}

        # Load the dataset
        df = pd.read_csv(DATA_FILE)

        # Convert the DataFrame to a JSON object
        data_json = df.to_dict(orient="records")

        return {"data": data_json}
    except Exception as e:
        return {"error": str(e)}
    
@router.get("/process-iris")
async def process_iris_route():
    """
    Processes the Iris dataset for training.
    """
    try:
        if not os.path.exists(DATA_FILE):
            return {"error": f"The dataset file was not found at {DATA_FILE}. Please download it first."}

        # Load the dataset
        df = load_iris_dataset2()

        # Process the dataset
        processed_df = process_iris_dataset(df)

        return {
            "processed_data": processed_df.to_dict(orient="records"),
            "message": "Data processed successfully!"
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/split-iris")
async def split_iris_route(test_size: float = 0.2):
    """
    Splits the Iris dataset into train and test sets.
    """
    try:
        if not os.path.exists(DATA_FILE):
            return {"error": f"The dataset file was not found at {DATA_FILE}. Please download it first."}

        # Load and process the dataset
        df = load_iris_dataset2()
        processed_df = process_iris_dataset(df)

        # Split the dataset
        X_train, X_test, y_train, y_test = split_iris_dataset(processed_df, test_size)

        return {
            "X_train": X_train.to_dict(orient="records"),
            "X_test": X_test.to_dict(orient="records"),
            "y_train": y_train.tolist(),
            "y_test": y_test.tolist(),
            "message": "Data split successfully!"
        }
    except Exception as e:
        return {"error": str(e)}
    
@router.get("/train-iris")
async def train_iris_model(test_size: float = 0.2):
    """
    Trains a RandomForestClassifier model using the Iris dataset.
    The dataset is loaded, processed, split, and then used for training.
    Args:
        test_size (float): Proportion of the dataset to include in the test split.
    Returns:
        dict: Training results and the path of the saved model.
    """
    try:
        # Ensure the dataset file exists
        if not os.path.exists(DATA_FILE):
            return {"error": f"The dataset file was not found at {DATA_FILE}. Please download it first."}
        
        # Step 1: Load the dataset
        df = load_iris_dataset2()
        
        # Step 2: Process the dataset
        processed_df = process_iris_dataset(df)
        
        # Step 3: Split the dataset
        X_train, X_test, y_train, y_test = split_iris_dataset(processed_df, test_size)
        
        # Step 4: Train the model using the helper function
        result = train_model(X_train, X_test, y_train, y_test)

        return result
    except Exception as e:
        return {"error": str(e)}
