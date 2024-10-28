import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Cargar los datos
data = pd.read_csv(r'data/all_stocks_5yr.csv', delimiter=',', on_bad_lines='skip')
print(data.shape)
print(data.sample(7))

data['date'] = pd.to_datetime(data['date'])
data.info()

close_data = data.filter(['close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .70))
print(training)

# Preparar características y etiquetas
x_train = []
y_train = []

for i in range(60, len(dataset)):
    x_train.append(dataset[i-60:i, 0])
    y_train.append(dataset[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Ajustar el scaler a los datos de entrenamiento
scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = scaler.fit_transform(x_train)

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train_scaled, x_test, y_train, y_test = train_test_split(x_train_scaled, y_train, test_size=0.2, random_state=42)

# Escalar x_test utilizando el mismo scaler ajustado
x_test_scaled = scaler.transform(x_test)

# Entrenar el modelo de XGBoost
model = xgb.XGBRegressor()
model.fit(x_train_scaled, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(x_test_scaled)

# Calcular el MSE y el RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Guardar las métricas de rendimiento en un archivo CSV
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE'],
    'Value': [mse, rmse]
})
metrics_df.to_csv(r'output/model_performance_metrics.csv', index=False)

# Guardar el modelo y el scaler
model.save_model(r'output/model.json')

with open('output/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)