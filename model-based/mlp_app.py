from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# 🔷 Load the trained pipeline
with open("mlp_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

app = FastAPI(title="Titanic Survival Prediction API")


# 🔷 Define input schema
class Passenger(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex: str         # "male" or "female"
    Embarked: str    # "C", "Q", "S"


@app.get("/")
def root():
    return {"message": "Titanic MLP Pipeline Prediction API is running 🚢"}


@app.post("/predict")
def predict_survival(passenger: Passenger):
    # 🔷 Build DataFrame
    df = pd.DataFrame([{
        "Pclass": passenger.Pclass,
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Fare": passenger.Fare,
        "Sex_male": 1 if passenger.Sex.lower() == "male" else 0,
        "Embarked_Q": 1 if passenger.Embarked.upper() == "Q" else 0,
        "Embarked_S": 1 if passenger.Embarked.upper() == "S" else 0,
    }])

    # 🔷 Predict
    pred = pipeline.predict(df)[0]
    proba = pipeline.predict_proba(df)[0]

    return {
        "survived": bool(pred),
        "probability": {
            "not_survived": round(proba[0], 4),
            "survived": round(proba[1], 4)
        }
    }
