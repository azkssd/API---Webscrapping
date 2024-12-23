from fastapi import APIRouter
import os
import opendatasets as od
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.services.data import load_iris_dataset2, process_iris_dataset, split_iris_dataset

router = APIRouter()

# Define the directory to save the dataset
DATA_DIR = "src/data"
# Define the path of the saved Iris.csv
DATA_FILE = os.path.join("src/data/iris/Iris.csv")

# Define the directory to save the trainning model
MODEL_DIR = os.path.join("src", "models")
# Define the path to the parameters for the model
CONFIG_PATH = os.path.join("src", "config", "model_parameters.json")

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
async def train_iris_model(data: dict):
    """
    Trains a RandomForestClassifier model on the provided dataset.
    Args:
        data (dict): The processed dataset in JSON format.
    Returns:
        dict: Training results and the path of the saved model.
    """
    try:
        # Convert JSON data to a DataFrame
        df = pd.DataFrame(data["data"])
        
        # Ensure the 'species' column exists (target variable)
        if 'species' not in df.columns:
            return {"error": "'species' column is required in the dataset."}
        
        # Split features (X) and target (y)
        X = df.drop(columns=['species'])
        y = df['species']
        
        # Split into training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Load model parameters
        if not os.path.exists(CONFIG_PATH):
            return {"error": f"Model configuration file not found at {CONFIG_PATH}."}
        
        with open(CONFIG_PATH, "r") as config_file:
            model_params = json.load(config_file)
        
        # Train the model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save the trained model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "iris_model.joblib")
        joblib.dump(model, model_path)
        
        return {
            "message": "Model trained and saved successfully!",
            "model_path": model_path,
            "accuracy": accuracy
        }
    except Exception as e:
        return {"error": str(e)}