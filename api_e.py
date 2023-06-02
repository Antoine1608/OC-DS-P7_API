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

@app.post("/predict")
def predict_credit(input:Input):
    dat = input.dict()
    data_in = df.loc[df['SK_ID_CURR']==dat['SK_ID_CURR'], L_var]
    prediction = best_model.predict(data_in)
    return{'prediction':prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
