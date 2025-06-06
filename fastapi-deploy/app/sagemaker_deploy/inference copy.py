import joblib
import os
import numpy as np
from typing import Any

# Called once when the model is loaded
def model_fn(model_dir: str) -> Any:
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    return model

# Called for every prediction request
def predict_fn(input_data, model) -> Any:
    # input_data is expected to be a dictionary of features
    features = np.array([list(input_data.values())])
    prediction = model.predict(features)
    return prediction.tolist()
