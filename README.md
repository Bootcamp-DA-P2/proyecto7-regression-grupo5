# 🏠 Sistema Avanzado de Predicción de Precios de Viviendas y Dashboard de Mercado

¡Bienvenido al repositorio central de nuestro proyecto grupal de Analítica de Datos y Machine Learning! Este proyecto abarca un pipeline completo de Ciencia de Datos: desde la ingesta de un dataset de alta dimensionalidad, pasando por una rigurosa auditoría de calidad de datos y optimización de modelos, hasta el despliegue de un modelo predictivo optimizado integrado en un dashboard interactivo moderno.

---

## 📦 Organización y Componentes del Proyecto

* **Modalidad:** Proyecto Grupal.
* **Plazo de Desarrollo:** 1 semana.
* **Componentes del Repositorio:** * Código fuente del pipeline de datos y modelado (`Eda.ipynb`).
  * Aplicación interactiva en producción (`dashboard.py`).
  * Recursos y artefactos del modelo serializados en rutas automatizadas (`data/utiles/`).
* **Solución Integrada:** La aplicación demo y la interfaz analítica de mercado se han consolidado de manera unificada en una sola herramienta interactiva utilizando **Streamlit** y **Plotly**, permitiendo realizar simulaciones de tasación y consultas de negocio desde el mismo entorno web.

---

## 📊 Origen de los Datos y Descripción General

Para este proyecto se ha utilizado el dataset estructurado en formato CSV llamado **`train.csv`**, el cual fue extraído de un set de datos de referencia en el sector inmobiliario de *Kaggle* ([House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)). El objetivo principal es modelar de forma precisa la variable numérica del precio de venta de propiedades (`SalePrice`) basándose en una amplia variedad de atributos estructurales, geográficos y de calidad.

El desarrollo se compone de dos módulos principales:
1. **Pipeline de Machine Learning (`Eda.ipynb`):** Notebook enfocado en el Análisis Exploratorio de Datos (EDA), manejo avanzado de valores nulos, eliminación de anomalías (*outliers*), comparación multimodelo y exportación serializada de recursos de producción.
2. **Dashboard Interactivo (`dashboard.py`):** Una aplicación web interactiva construida sobre **Streamlit** y **Plotly** que actúa como interfaz para el usuario final, permitiendo cotizar viviendas en tiempo real y visualizar analíticas de mercado.

---

## ⚙️ Metodología y Alcance Técnico del Pipeline

El diseño del proyecto implementa metodologías avanzadas de ingeniería de características, validación estadística y analítica de datos en múltiples niveles de profundidad:

### 1. Auditoría y Calidad de Datos (Preprocesamiento)
* **Tratamiento de Valores Faltantes:** Columnas cualitativas donde el nulo representa la ausencia física de una amenidad (`PoolQC`, `GarageType`, etc.) se imputan con la cadena `'None'`. Las variables numéricas de áreas se inicializan en `0`. El campo `LotFrontage` se completa mediante la **mediana** para evitar sesgos por asimetría, y `Electrical` mediante su **moda**.
* **Tratamiento de Outliers:** Para blindar la estabilidad de los algoritmos frente a distorsiones extremas, se remueven las viviendas con superficies habitables aéreas (`GrLivArea`) anómalas (superiores a los 4,000 pies cuadrados), estabilizando las pendientes generales de regresión.

### 2. Entrenamiento, Robustez y Evaluación del Modelo
* **Ingeniería de Características:** Se escalan las variables numéricas mediante un `StandardScaler` y las cualitativas se transforman mediante `OneHotEncoder` para asegurar que el modelo interprete correctamente las zonas geográficas y calidades.
* **Estrategia Multimodelo y Ensembles:** Se evalúan y comparan tanto modelos lineales regularizados (**Ridge** y **Lasso**) como algoritmos basados en árboles y ensambles avanzados (**Random Forest Regressor** y **XGBoost Regressor**).
* **Validación Cruzada e Hiperparámetros:** El proceso incluye técnicas de validación cruzada para certificar la capacidad de generalización del modelo en datos no vistos, minimizando el riesgo de *overfitting*. A su vez, se aplican estrategias de optimización y ajuste sistemático de hiperparámetros para exprimir la precisión de los algoritmos.
* **Métricas de Rendimiento:** La evaluación se realiza bajo un estándar riguroso de métricas de regresión (**RMSE, MAE, R²**), complementado con el análisis de residuos y contrastes visuales de valores reales frente a predicciones.

---

## 🖥️ Arquitectura del Dashboard de Producción

La interfaz interactiva se divide en dos grandes bloques estratégicos que guían al usuario desde una perspectiva macro (visión global de mercado) hacia una micro (tasación de una vivienda específica):

### SECCIÓN 1: EXPLORADOR (Enfoque Macro de Mercado)
Diseñado para entender el comportamiento histórico e identificar las lógicas del negocio inmobiliario mediante visualizaciones dinámicas de Plotly:
* **Evolución Histórica (`fig_evol`):** Incorpora la banda de desviación estándar y la mediana para reflejar de forma transparente la volatilidad y la dispersión de los precios año a año.
* **Scatter Plot (`fig_scatter`):** Gráfico clave para validar de forma visual las hipótesis lineales de regresión (ej. relación entre metros cuadrados y coste) y descubrir clústeres de propiedades cambiando los ejes de forma interactiva.
* **Gráfico de Tarta/Donut (`fig_pie`):** Muestra el volumen de la muestra implementando un filtro inteligente que agrupa categorías con representación menor al 2% bajo la etiqueta "Otros", evitando la saturación del gráfico.
* **Heatmap (`fig12`):** Matriz de correlación que sirve como antesala al Machine Learning, demostrando matemáticamente qué características físicas tienen el impacto más fuerte sobre el valor final.

### SECCIÓN 2: PREDICTOR (Enfoque Micro y Simulación en Vivo)
Interactúa con los archivos binarios serializados (`.pkl`) generados por el pipeline, transformando los inputs del usuario mediante componentes que abren la "caja negra" del algoritmo:
* **Barras de Coeficientes (`fig_contrib`):** Desglosa el porqué matemático de la tasación calculada por el modelo Ridge, separando visualmente qué atributos suman valor a la vivienda y cuáles restan o influyen menos.
* **Barras Comparativo (`fig_comp`):** Segmenta el dataset original en tiempo real para localizar las 5 viviendas reales más parecidas en tamaño y calidad a la simulada, situando la predicción al lado para validar su coherencia comercial.
* **Histograma de Densidad (`fig_dist`):** Genera la campana de distribución de precios del mercado real y cruza una línea vertical de corte que sitúa la predicción del usuario, permitiendo identificar al instante el posicionamiento de la vivienda en el mercado total.

---

## 🛠️ Tecnologías Empleadas

* **Entornos de Trabajo:** Jupyter Notebook, Kaggle Notebooks.
* **Procesamiento de Datos:** NumPy, Pandas, SciPy.
* **Modelado y Machine Learning:** Scikit-Learn (sklearn), XGBoost, Optuna/Search-Tuning para hiperparámetros.
* **Visualización Gráfica:** Matplotlib (Pyplot), Seaborn, Plotly Express y Plotly Graph Objects.
* **Despliegue e Interfaz UI:** Streamlit Framework.
* **Serialización de Recursos:** Joblib.

---

## 📁 Estructura del Repositorio

El proyecto implementa una estructura modular optimizada para garantizar la consistencia entre la etapa de entrenamiento y la fase de inferencia web:

```text
├── data/
│   ├── train.csv                      # Dataset original (extracción de Kaggle)
│   └── utiles/                        # Directorio automatizado de recursos compartidos
│       ├── clean_data/
│       │   └── clean_train.csv        # Dataset tras auditoría e imputación de nulos
│       ├── outliers/
│       │   └── data_con_outliers_marcados.csv
│       └── modelo/
│           ├── modelo_ridge_house_prices.pkl   # Modelo ganador serializado
│           ├── escalador_house_prices.pkl      # StandardScaler ajustado en train
│           └── columnas_modelo.pkl             # Listado de columnas tras One-Hot Encoding
├── Eda.ipynb                            # Pipeline completo de entrenamiento, EDA y ML
├── Readme.txt                       # Código fuente de la aplicación interactiva Streamlit
└── dashboard.py                          # Documentación del proyecto (este archivo)