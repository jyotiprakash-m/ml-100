from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load the model using pickle
with open("power_transformation_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the input schema
class StrengthInput(BaseModel):
    Cement: float
    Blast_Furnace_Slag: float
    Fly_Ash: float
    Water: float
    Superplasticizer: float
    Coarse_Aggregate: float
    Fine_Aggregate: float
    Age: int

@app.post("/predict")
def predict_strength(data: StrengthInput):
    # Prepare input for the model
    input_array = np.array([[  # Ensure 2D array
        data.Cement,
        data.Blast_Furnace_Slag,
        data.Fly_Ash,
        data.Water,
        data.Superplasticizer,
        data.Coarse_Aggregate,
        data.Fine_Aggregate,
        data.Age
    ]])

    # Predict
    prediction = model.predict(input_array)[0]

    return {"prediction": prediction}
