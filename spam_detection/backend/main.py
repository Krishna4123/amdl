from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np

app = FastAPI()

# Load model & vectorizer
# Using try-except block to handle missing files gracefully during initial setup
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("Model and Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    vectorizer = None

class Message(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Spam Detection API is running. Send POST request to /predict"}

@app.post("/predict")
def predict_spam(data: Message):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure model.pkl and vectorizer.pkl are present.")
    
    try:
        message = [data.text]
        vector = vectorizer.transform(message)
        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0][1]

        return {
            "prediction": int(prediction),
            "ham_probability": float(prob),
            "spam_probability": float(1 - prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
