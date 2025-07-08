from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Load the model
with open('titanic_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

class Passenger(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex: str        # e.g., "male" or "female"
    Embarked: str   # e.g., "C", "Q", or "S"
@app.get("/")
def read_root():
    return {"message": "Titanic Prediction API is up!"}

@app.post("/predict")
def predict(passenger: Passenger):
    # Create input dataframe
    data = pd.DataFrame([{
        "Pclass": passenger.Pclass,
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Fare": passenger.Fare,
        "Sex": passenger.Sex,
        "Embarked": passenger.Embarked
    }])

    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]

    return {
        "survived": int(prediction),
        "probability": {
            "not_survived": round(probabilities[0], 4),
            "survived": round(probabilities[1], 4)
        }
    }
