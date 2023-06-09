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
import shap

# Initialisation de l'application FastAPI
app = FastAPI()

# Charger les données
df = pd.read_csv("dfex.csv")
X = pd.read_csv("X.csv")

# Définir la première colonne en tant qu'index
X = X.set_index(X.iloc[:, 0])

# Supprimer la première colonne du DataFrame
X = X.iloc[:, 1:]

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

# Charger Explainer
with open('exp.pickle', 'rb') as file:
    exp = pickle.load(file)

class Input(BaseModel):
    SK_ID_CURR:int

@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API du projet 7 - Implémentez un modèle de scoring"}

@app.post("/predict")
def predict_credit(input:Input):
    dat = input.dict()
    data_in = df.loc[df['SK_ID_CURR']==dat['SK_ID_CURR'], L_var]
    prediction = best_model.predict_proba(data_in)
    
    if prediction[0][1] <= best_th:
        result = {
            "prediction": f"Crédit accordé (prédiction <= {best_th})",
            "risque_defaut": round(prediction[0][1], 2)
        }
    else:
        result = {
            "prediction": f"Crédit refusé (prédiction > {best_th})",
            "risque_defaut": round(prediction[0][1], 2)
        }

    return result

'''@app.post("/graphe")
def expl(input:Input):
    don = input.dict()
    num = don['SK_ID_CURR']
    #Shap client
    idx = df[df['SK_ID_CURR'] == num].index.item()
    exp_cust = exp[idx]
    

    #Shap global
    idx = X.index.get_loc('mean')
    exp_glob = exp[idx]
    # Convertir l'objet exp_cust en un dictionnaire JSON-compatible
    expl_glob = {
        'values': exp_glob.values.tolist(),
        'base_values': exp_glob.base_values.tolist(),
        'data': exp_glob.data.tolist(),
        # Ajoutez d'autres attributs pertinents ici
    }
    
    # Sérialiser le dictionnaire en JSON
    json_xg = jsonable_encoder(expl_glob)
    
    # Retourner la réponse HTTP avec l'explication sérialisée en JSON
    return json_xg 

    #Shap similaire
    # fonction pour récupérer l'âge d'un client
    def roundDown(n):
        a=int(-n/3640)
        return 10*a
    sex = int(df.loc[df['SK_ID_CURR']== num, 'CODE_GENDER'])
    age = int(df.loc[df['SK_ID_CURR']== num, 'DAYS_BIRTH'])

    index = 's' + str(sex) + 'm' + str(roundDown(age))

    idx = X.index.get_loc(index)

    exp_sim = exp[idx]
    # Convertir l'objet exp_cust en un dictionnaire JSON-compatible
    expl_sim = {
        'values': exp_sim.values.tolist(),
        'base_values': exp_gsim.base_values.tolist(),
        'data': exp_sim.data.tolist(),
        # Ajoutez d'autres attributs pertinents ici
    }
    
    # Sérialiser le dictionnaire en JSON
    json_xs = jsonable_encoder(expl_sim)
    
    # Retourner la réponse HTTP avec l'explication sérialisée en JSON
    return json_xs

    return exp_cust'''

from fastapi.encoders import jsonable_encoder

@app.post("/graphe")
def expl(input: Input):
    dat = input.dict()
    num = dat['SK_ID_CURR']

    #Shap customer
    # Obtenir l'index correspondant à SK_ID_CURR
    idx = df[df['SK_ID_CURR'] == num].index.item()
    
    # Obtenir l'explication du client
    exp_cust = exp[idx]
    
    # Convertir l'objet exp_cust en un dictionnaire JSON-compatible
    expl_cust = {
        'values': exp_cust.values.tolist(),
        'base_values': exp_cust.base_values.tolist(),
        'data': exp_cust.data.tolist(),
        # Ajoutez d'autres attributs pertinents ici
    }
    
    # Sérialiser le dictionnaire en JSON
    json_xc = jsonable_encoder(expl_cust)
    
    # Retourner la réponse HTTP avec l'explication sérialisée en JSON
    #return json_xc

    #Shap global
    idx = X.index.get_loc('mean')
    exp_glob = exp[idx]
    # Convertir l'objet exp_cust en un dictionnaire JSON-compatible
    expl_glob = {
        'values': exp_glob.values.tolist(),
        'base_values': exp_glob.base_values.tolist(),
        'data': exp_glob.data.tolist(),
        # Ajoutez d'autres attributs pertinents ici
    }
    
    # Sérialiser le dictionnaire en JSON
    json_xg = jsonable_encoder(expl_glob)
    
    # Retourner la réponse HTTP avec l'explication sérialisée en JSON
    #return json_xg 

    #Shap similaire
    # fonction pour récupérer l'âge d'un client
    def roundDown(n):
        a=int(-n/3640)
        return 10*a
    sex = int(df.loc[df['SK_ID_CURR']== num, 'CODE_GENDER'])
    age = int(df.loc[df['SK_ID_CURR']== num, 'DAYS_BIRTH'])

    index = 's' + str(sex) + 'm' + str(roundDown(age))

    idx = X.index.get_loc(index)

    exp_sim = exp[idx]
    # Convertir l'objet exp_cust en un dictionnaire JSON-compatible
    expl_sim = {
        'values': exp_sim.values.tolist(),
        'base_values': exp_sim.base_values.tolist(),
        'data': exp_sim.data.tolist(),
        # Ajoutez d'autres attributs pertinents ici
    }
    
    # Sérialiser le dictionnaire en JSON
    json_xs = jsonable_encoder(expl_sim)
    
    # Retourner la réponse HTTP avec l'explication sérialisée en JSON
    #return json_xs
    return json_xc#, json_xg, json_xs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
