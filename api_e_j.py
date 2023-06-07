import matplotlib
import matplotlib.pyplot as plt
import json
from typing import List
from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import uvicorn
from pydantic import BaseModel

# Initialisation de l'application FastAPI
app = FastAPI()

# Charger les données
df = pd.read_csv("dfex.csv")

# Charger les variables threshold et important features
# Opening JSON file
f = open('data.json')
  
# returns JSON object as a dictionary
data = json.load(f)

# Charger le meilleur seuil
best_th = data['best_th']

# Charger la liste des features importantes
L_var = data['feat']

# Charger le meilleur modèle
best_model = pickle.load(open('model.pkl', 'rb'))

class Input(BaseModel):
    SK_ID_CURR:int

@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API du projet 7 - Implémentez un modèle de scoring"}

@app.get("/predict/{num}")
def predict(num: int):
    
    data = df.loc[df['SK_ID_CURR']==num, L_var].values

    y_te_pred = best_model.predict(data)
    y_te_pred = (y_te_pred >= best_th)
        
    y_proba = best_model.predict_proba(data)
    proba = y_proba[0][1]

    if proba <= best_th:
        result = {
            "prediction": "Crédit accordé",
            "risque_defaut": round(proba, 2)
        }
    else:
        result = {
            "prediction": "Crédit refusé",
            "risque_defaut": round(proba, 2)
        }
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
