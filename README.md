# Trabajo Final MLOps MDS 2023 Model Deployment

Este repositorio contiene una aplicación FastAPI para servir un modelo de machine learning guardado. La aplicación está configurada para ejecutarse en un contenedor Docker.

## Integrantes
- Nicole Orellana
- José Fernández

## Requisitos

- Docker
- Docker Compose

## Estructura del Proyecto
```
project-root 
│ 
├── app.py 
├── requirements.txt 
└── Dockerfile 
```

## How to

### 1. Clonar el Repositorio

```sh
git clone https://github.com/belta1/mlops_trabajofinal.git
cd mlops_trabajofinal
```

### 2. Descargar la imagen desde dockerhub
```sh
docker pull belta1/stock_prediction_xgb:latest
```

### 3. Ejecutar el contenedor con la imagen descargada
```sh
docker run -d -p 8000:8000 --name stock_prediction_xgb belta1/stock_prediction_xgb:latest
```