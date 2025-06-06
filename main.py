from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# ----------------------------------------
# Logging setup
# ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------------------
# Load model
# ----------------------------------------
try:
    model = joblib.load("model.pkl")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model.")
    raise e

# ----------------------------------------
# Define input schema
# ----------------------------------------
class UPDRSPredictRequest(BaseModel):
    age: float
    sex: float
    test_time: float
    Jitter_percent: float
    Jitter_Abs: float
    Jitter_RAP: float
    Jitter_PPQ: float
    Jitter_DDP: float
    Shimmer: float
    Shimmer_dB: float
    Shimmer_APQ3: float
    Shimmer_APQ5: float
    Shimmer_APQ11: float
    Shimmer_DDA: float
    NHR: float
    HNR: float
    RPDE: float
    DFA: float
    PPE: float

# ----------------------------------------
# Initialize app
# ----------------------------------------
app = FastAPI()

@app.get("/ping")
def ping():
    logger.info("Health check endpoint called.")
    return {"message": "pong"}

@app.post("/predict")
def predict(data: UPDRSPredictRequest):
    try:
        # Log input
        logger.info(f"Received prediction request: {data.json()}")

        # Convert input to model format
        input_array = np.array([[ 
            data.age, data.sex, data.test_time,
            data.Jitter_percent, data.Jitter_Abs, data.Jitter_RAP,
            data.Jitter_PPQ, data.Jitter_DDP,
            data.Shimmer, data.Shimmer_dB, data.Shimmer_APQ3,
            data.Shimmer_APQ5, data.Shimmer_APQ11, data.Shimmer_DDA,
            data.NHR, data.HNR, data.RPDE, data.DFA, data.PPE
        ]])

        # Make prediction
        prediction = model.predict(input_array)
        logger.info(f"Prediction successful: {prediction[0]}")

        return {"predicted_total_UPDRS": prediction[0]}
    
    except Exception as e:
        logger.exception("Error during prediction.")
        return {"error": "Prediction failed. Please check input data and server logs."}
