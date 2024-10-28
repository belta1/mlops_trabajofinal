from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

# Inicializar la aplicación FastAPI
app = FastAPI()

# Definir el modelo de datos para la solicitud
class PredictionRequest(BaseModel):
    data: List[List[float]]

# Cargar el modelo y el scaler desde los archivos
model = xgb.XGBRegressor()
model.load_model('output/model.json')

with open('output/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Definir la ruta para las predicciones
@app.post("/predict")
def predict(request: PredictionRequest):
    # Convertir los datos de entrada a un array numpy
    input_data = np.array(request.data)
   
    # Verificar la forma de los datos de entrada
    if input_data.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Expected input with {scaler.n_features_in_} features, but got {input_data.shape[1]} features.")
    
    # Escalar los datos de entrada
    input_data_scaled = scaler.transform(input_data)
    
    # Realizar predicciones
    predictions = model.predict(input_data_scaled)
    
    # Convertir las predicciones a una lista
    predictions_list = predictions.tolist()
    
    return {"predictions": predictions_list}

# Ruta de prueba
@app.get("/")
def read_root():
    return {"message": "API is running"}

# Ejecutar la aplicación con reinicio automático
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)