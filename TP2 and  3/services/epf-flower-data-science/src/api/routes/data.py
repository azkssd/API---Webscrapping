from fastapi import APIRouter
import os
import opendatasets as od

router = APIRouter()

# Define the directory to save the dataset
DATA_DIR = "src/data"

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
