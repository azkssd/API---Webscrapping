import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Define constants for file paths
DATA_DIR = os.path.join("src", "data")
DATA_FILE = os.path.join(DATA_DIR, "iris", "Iris.csv")

# Define the directory to save the trainning model
MODEL_DIR = os.path.join("src", "models")
# Define the path to the parameters for the model
CONFIG_PATH = os.path.join("src", "config", "model_parameters.json")

def load_iris_dataset2():
    """
    Loads the Iris dataset from the saved CSV file.
    Returns:
        DataFrame: The loaded dataset as a pandas DataFrame.
    """
    # Load the dataset
    df = pd.read_csv(DATA_FILE)
    return df


def process_iris_dataset(df):
    """
    Processes the Iris dataset for training.
    Args:
        df (DataFrame): The raw Iris dataset.
    Returns:
        DataFrame: The processed dataset.
    """
    # Drop missing values
    df = df.dropna()

    # Encode the target variable (species)
    #df['species'] = df['Species'].astype('category').cat.codes

    # Encode the target variable (species)
    label_encoder = LabelEncoder()
    df['species'] = label_encoder.fit_transform(df['Species'])
    return df


def split_iris_dataset(df, test_size=0.2):
    """
    Splits the Iris dataset into training and test sets.
    Args:
        df (DataFrame): The processed Iris dataset.
        test_size (float): Proportion of the dataset to include in the test split.
    Returns:
        tuple: X_train, X_test, y_train, y_test datasets.
    """
    # Separate features and labels
    X = df.drop(columns=['species', 'Species', 'Id'])  # Exclude species and the other old Species
    y = df['species']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    """
    Trains a RandomForestClassifier using provided training and testing data.
    Args:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.
    Returns:
        dict: Contains the accuracy, model path, and other details.
    """
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "iris_model.joblib")

    # Load model parameters
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Model configuration file not found at {CONFIG_PATH}.")

    with open(CONFIG_PATH, "r") as config_file:
        model_params = json.load(config_file)

    # Train the model
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the trained model
    joblib.dump(model, model_path)

    return {
        "accuracy": accuracy,
        "model_path": model_path,
        "message": "Model trained and saved successfully!"
    }