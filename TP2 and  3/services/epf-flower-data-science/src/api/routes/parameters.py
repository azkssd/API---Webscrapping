from fastapi import FastAPI, APIRouter, HTTPException

import firebase_admin
from firebase_admin import credentials, firestore
from fastapi.responses import JSONResponse

from pydantic import BaseModel

router = APIRouter()

# Path to the Firebase credentials file
CRED_PATH = 'firebase_credentials.json'
cred = credentials.Certificate(CRED_PATH)

# Initialize the Firebase Admin SDK
firebase_admin.initialize_app(cred)
# Get Firestore client
db = firestore.client()

# Route to get parameters
@router.get('/get-parameters')
def get_parameters():
    try:
        # Reference to the 'parameters' document in the 'parameters' collection
        doc_ref = db.collection('parameters').document('parameters')
        
        # Retrieve the document data
        doc = doc_ref.get()
        
        if doc.exists:
            parameters = doc.to_dict()  # Get data as dictionary
            return JSONResponse(content=parameters, status_code=200)
        else:
            return JSONResponse(content={"error": "Parameters document not found"}, status_code=404)
            
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# Create a Pydantic model for the request body
class Parameters(BaseModel):
    n_estimators: int
    criterion: str

# Route to update parameters
@router.post("/update-parameters")
async def update_parameters(params: Parameters):
    try:
        # Reference to the 'parameters' document in the 'parameters' collection
        doc_ref = db.collection('parameters').document('parameters')
        
        # Check if the document exists
        doc = doc_ref.get()
        
        if doc.exists:
            # Update the document with new values
            doc_ref.update({
                "n_estimators": params.n_estimators,
                "criterion": params.criterion
            })
            return JSONResponse(content={"message": "Parameters updated successfully"}, status_code=200)
        else:
            raise HTTPException(status_code=404, detail="Parameters document not found")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Route to add parameters
@router.put("/add-parameters")
async def add_parameters(params: Parameters):
    try:
        # Reference to the 'parameters' document in the 'parameters' collection
        doc_ref = db.collection('parameters').document('parameters')

        # Check if the document exists
        doc = doc_ref.get()
        
        if not doc.exists:
            # Create a new document with the provided parameters if it doesn't exist
            doc_ref.set({
                "n_estimators": params.n_estimators,
                "criterion": params.criterion
            })
            return JSONResponse(content={"message": "Parameters added successfully"}, status_code=201)
        else:
            raise HTTPException(status_code=400, detail="Parameters document already exists")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)