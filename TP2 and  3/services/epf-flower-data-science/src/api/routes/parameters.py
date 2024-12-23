from fastapi import APIRouter

import firebase_admin
from firebase_admin import credentials, firestore
from flask import jsonify

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
            return jsonify(parameters), 200
        else:
            return jsonify({"error": "Parameters document not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500