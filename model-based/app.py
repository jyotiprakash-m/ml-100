from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# load model & scaler
with open("fare_model.pkl", "rb") as f:
    scaler, model = pickle.load(f)

app = FastAPI(title="Titanic Fare Prediction API")


# Input schema
class Passenger(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Sex_male: int         # 0 or 1
    Embarked_Q: int       # 0 or 1
    Embarked_S: int       # 0 or 1


@app.get("/")
def root():
    return {"message": "Titanic Fare Prediction API is running!"}


@app.post("/predict")
def predict_fare(passenger: Passenger):
    # create dataframe from input
    data = pd.DataFrame([{
        "Pclass": passenger.Pclass,
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Sex_male": passenger.Sex_male,
        "Embarked_Q": passenger.Embarked_Q,
        "Embarked_S": passenger.Embarked_S
    }])

    # scale

    X_scaled = scaler.transform(data)

    # predict
    fare = max(0, model.predict(X_scaled)[0])

    return {
        "predicted_fare": round(fare, 2)
    }
