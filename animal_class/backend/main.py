from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from preprocess import extract_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
log_reg = joblib.load("models/log_reg.joblib")
# lin_reg = joblib.load("models/lin_reg.joblib")  # Broken model
nb = joblib.load("models/nb.joblib")
knn = joblib.load("models/knn.joblib")
class_names = joblib.load("models/class_names.joblib")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Convert image → features
    features = extract_features(file.file)

    # Expand dims for sklearn
    X = np.expand_dims(features, 0)

    # Logistic Regression (probabilities)
    log_proba = log_reg.predict_proba(X)[0]
    log_output = {class_names[i]: float(log_proba[i]) for i in range(len(class_names))}

    # Linear Regression (convert output → softmax)
    # lin_output_raw = lin_reg.predict(X).reshape(-1)
    # lin_proba = softmax(lin_output_raw)
    # lin_output = {class_names[i]: float(lin_proba[i]) for i in range(len(class_names))}
    lin_output = {}  # Empty dict for disabled model

    # Naive Bayes
    nb_proba = nb.predict_proba(X)[0]
    nb_output = {class_names[i]: float(nb_proba[i]) for i in range(len(class_names))}

    # KNN 
    knn_proba = knn.predict_proba(X)[0]
    knn_output = {class_names[i]: float(knn_proba[i]) for i in range(len(class_names))}

    # Combine models → choose highest overall confidence
    all_outputs = {
        "logistic_regression": log_output,
        "linear_regression": lin_output,
        "naive_bayes": nb_output,
        "knn": knn_output
    }

    # Find best model prediction
    best_model = None
    best_class = None
    best_conf = -1

    for model_name, probs in all_outputs.items():
        for cls, prob in probs.items():
            if prob > best_conf:
                best_conf = prob
                best_model = model_name
                best_class = cls

    return {
        "best_model": best_model,
        "best_prediction": best_class,
        "confidence": float(best_conf),
        "all_model_outputs": all_outputs
    }
