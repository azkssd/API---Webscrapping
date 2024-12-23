import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define constants for file paths
DATA_DIR = os.path.join("src", "data")
DATA_FILE = os.path.join(DATA_DIR, "iris", "Iris.csv")


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
    df['species'] = df['Species'].astype('category').cat.codes
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
    X = df.drop(columns=['species'])
    y = df['species']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test