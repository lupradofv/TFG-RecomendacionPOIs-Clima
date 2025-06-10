# TFG: Impacto del Clima en la Recomendación de Puntos de Interés

Este repositorio contiene todo el código, los datos procesados y la documentación asociados al Trabajo de Fin de Grado (TFG) titulado:

**"Impacto de la Información Meteorológica en la Recomendación de Puntos de Interés"**

Autora: Lucía Prado Fernández-Vega  
Universidad Pontificia Comillas - ICAI  
Grado en Ingeniería Matemática e Inteligencia Artificial  
Curso 2024–2025

---

## 📚 Descripción

Este proyecto analiza cómo las condiciones meteorológicas (temperatura, lluvia, viento, etc.) afectan a los sistemas de recomendación contextuales de Puntos de Interés (POIs). Se implementan modelos tradicionales y avanzados de recomendación con y sin contexto climático para evaluar su rendimiento.

---

## 🗂️ Estructura del repositorio

```bash

TFG-RecomendacionPOIs-Clima/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── FoursquareGlobalCheckinDataset/    # Datos crudos originales
│   ├── FoursquareProcessed/               # Datos procesados por ciudad
│   └── weather/                           # Datos meteorológicos

├── data-processing/
│   ├── foursquare-processing.ipynb          # Procesamiento de ciudades, POIs y check-ins
│   └── weather-processing.ipynb           # Procesamiento de datos climáticos

└── models/
    ├── baselines/                         # Recomendadores base: aleatorio, popularidad
    │   ├── popularity_recommender.py
    │   └── random_recommender.py
    ├── knn/                               # KNN clásico y con clima
    │   └── ...
    └── mf/                                # Factorización de matrices (SGD, re-rank, embeddings clima)
        └── ...
   
```

---
## ⚙️ Requisitos

Este proyecto usa **Python 3.10+** y las siguientes librerías:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `torch`

Instálalas con:

```bash
pip install -r requirements.txt
```

---

## 📊 Modelos desarrollados

Los siguientes sistemas de recomendación han sido implementados:

- 🔹 **Recomendador aleatorio**

- 🔹 **Recomendador por popularidad**

- 🔹 **K-Nearest Neighbors (KNN)**
  - Versión clásica
  - Versión con contexto climático (similitud combinada)

- 🔹 **Factorización de matrices**
  - Clásica con **Stochastic Gradient Descent (SGD)**
  - Con contexto climático mediante:
    - **Ponderación** de las interacciones según clima
    - **Reordenamiento** de recomendaciones según perfil climático
  - Con **Deep Learning** en PyTorch usando embeddings climáticos

---

## 📈 Evaluación

Las métricas utilizadas para evaluar el rendimiento de los modelos incluyen:

- `Precision@10`: proporción de POIs relevantes entre los 10 recomendados
- `Recall@10`: proporción de POIs relevantes recuperados entre los relevantes totales
- `Expected Popularity Complement (EPC)`: mide la novedad recomendada
- `Aggregate Diversity@10`: número total de POIs distintos recomendados a todos los usuarios

Estas métricas fueron calculadas de forma separada en tres ciudades:

- **Londres**
- **Nueva York**
- **Tokio**

---

## 🧑‍💻 Autoría

**Lucía Prado Fernández-Vega**  
TFG dirigido por **Pablo Sánchez Pérez**  
Grado en **Ingeniería Matemática e Inteligencia Artificial**  
**Universidad Pontificia Comillas**

