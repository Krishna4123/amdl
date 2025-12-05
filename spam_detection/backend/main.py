from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://amdl-one.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model safely
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model files:", e)
    model = None
    vectorizer = None


class Message(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Spam Detection API is running"}


@app.post("/predict")
def predict_spam(data: Message):
    if model is None or vectorizer is None:
         return {"error": "Model not loaded properly on server."}

    message = [data.text]
    vector = vectorizer.transform(message)
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0][1]
    
    return {
        "prediction": int(pred),
        "ham_probability": float(prob),
        "spam_probability": float(1 - prob)
    }

