'''
Creation of the collection 'parameters' in the 'Data Sources' firestore project 
'''

import firebase_admin
from firebase_admin import credentials, firestore

# Path to the Firebase credentials file
CRED_PATH = 'firebase_credentials.json'
cred = credentials.Certificate(CRED_PATH)

# Initialize the Firebase Admin SDK
firebase_admin.initialize_app(cred)

parameters = {
    'n_estimators': 100,  # Example
    'criterion': 'gini'   # Example
}

# Get a Firestore client
db = firestore.client()

db.collection('parameters').document('parameters').set(parameters)
print("Parameters have been written to Firestore.")
