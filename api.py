# Import des librairies uvicorn, pickle, FastAPI, File, UploadFile, BaseModel
from fastapi import FastAPI, File, UploadFile, Depends, Body
import uvicorn
import numpy as np 
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import mlflow
import os
import boto3

# Création des tags
tags = [
       {
              "name": "Hello name V1",
              "description": "Endpoint pour dire bonjour avec un nom en paramètre.",
       },
       {
              "name": "Predict V1",
              "description": "Endpoint pour prédire un résultat à partir du modèle 1 avec plusieurs caractéristiques.",
       },
       {
              "name": "Predict V2",
              "description": "Endpoint pour prédire un résultat à partir du modèle 2 avec moins de caractéristiques.",
       },
]

# Création de l'application
app = FastAPI(
       title="API de prediction",
       description="Predictions",
       version= "1.0.0",
       openapi_tags= tags
)

# Chargement du modèle
model_pkl_file = "model_1.pkl"
with open(model_pkl_file, 'rb') as file:
    svm_model = pickle.load(file)

model_pkl_file = "model_2.pkl"
with open(model_pkl_file, 'rb') as file:
    svm_model_2 = pickle.load(file)

# Chargement des LabelEncoder
label_encoder_file = "labelencoder.pkl"
with open(label_encoder_file, 'rb') as file:
    loaded_data = pickle.load(file)
le_sleep = loaded_data["le_sleep"]
le_gender = loaded_data["le_gender"]

# Point de terminaison avec paramètre
@app.get("/hello", tags=["Hello name V1"])
def hello(name: str='World'):
    """
    Endpoint permettant de saluer avec un message personnalisé.

    Args:
       name (str): Le nom à saluer. Par défaut, 'World'.

    Returns:
       dict: Un dictionnaire contenant le message de salutation.
    """
    return {"message": f"Hello {name}"}

# Création du modèle de données pour le modéle 1 ('Gender', 'Age', 'Physical Activity Level', 'Heart Rate', 'Daily Steps', 'BloodPressure_high', 'BloodPressure_low', 'Sleep Disorder'])
class Credit(BaseModel):
    Gender : str
    Age : int
    Physical_Activity_Level : int
    Heart_Rate : int
    Daily_Steps : int
    BloodPressure_high : int
    BloodPressure_low : int
        
# Point de terminaison : Prédiction 1
@app.post("/predict", tags=["Predict V1"])
def predict(credit: Credit):
    """
    Args:

       - Gender (str): 'Male' ou 'Female'

       - Age (int): Age de la personne.

       - Physical_Activity_Level:  Niveau d'activité physique.

       - Heart_Rate : Fréquence cardiaque.

       - Daily_Steps : Nombre de pas effectué par jour.

       - BloodPressure_high : La pression artérielle systolique.

       - BloodPressure_low : La pression artérielle diastolique.

       _______________________________________________________________________  
    Returns:
       dict: Un dictionnaire contenant le résultat de la prédiction.
    """
    gender_numerical = le_gender.transform([credit.Gender])[0]
    
    X_new = [[gender_numerical, credit.Age, credit.Physical_Activity_Level,
              credit.Heart_Rate, credit.Daily_Steps,
              credit.BloodPressure_high, credit.BloodPressure_low]]
    
    prediction_result = svm_model.predict(X_new)
    prediction_result_categorical = le_sleep.inverse_transform([prediction_result])
    return {"prediction": prediction_result_categorical.tolist()}

# Création du modèle de données pour le modéle 2 ('Physical Activity Level', 'Heart Rate', 'Daily Steps', 'Sleep Disorder')
class Credit_2(BaseModel):
    Physical_Activity_Level : int
    Heart_Rate : int
    Daily_Steps : int

# Point de terminaison : Prédiction 2
@app.post("/predict2", tags=["Predict V2"])
def predict2(credit: Credit_2):
    """
    Args:

       - Physical_Activity_Level:  Niveau d'activité physique.

       - Heart_Rate : Fréquence cardiaque.

       - Daily_Steps : Nombre de pas effectué par jour.

       _______________________________________________________________________  
    Returns:
       dict: Un dictionnaire contenant le résultat de la prédiction.
    """
    X_new = [[credit.Physical_Activity_Level, credit.Heart_Rate, credit.Daily_Steps]]
    
    prediction_result = svm_model_2.predict(X_new)
    prediction_result_categorical = le_sleep.inverse_transform([prediction_result])
    return {"prediction": prediction_result_categorical.tolist()}

# Démarage de l'application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)