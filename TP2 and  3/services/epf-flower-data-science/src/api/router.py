"""API Router for Fast API."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import joblib
import numpy as np

from src.api.routes import hello
from src.api.routes import data

router = APIRouter()

router.include_router(hello.router, tags=["Hello"])
router.include_router(data.router, tags=["Data"])

# Define the prediction data model
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

MODEL_PATH = os.path.join("src", "models", "iris_model.joblib")

@router.post("/predict-iris")
async def predict_iris(input_data: PredictionInput):
    """
    Predict the species of an Iris flower based on its features.
    Args:
        input_data (PredictionInput): Features for prediction.
    Returns:
        dict: Predicted species as JSON.
    """
    # Ensure the model exists
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Trained model not found.")

    # Load the model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")

    # Extract input features as a numpy array (expects a 2D array)
    features = np.array([[input_data.sepal_length,
                           input_data.sepal_width,
                           input_data.petal_length,
                           input_data.petal_width]])

    # Make prediction
    try:
        prediction = model.predict(features)
        species = prediction[0]  # Assuming single prediction
        return {"predicted_species": int(species)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")