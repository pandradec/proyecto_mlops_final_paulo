# MLOps Introduction: Final Project

proyecto_mlops_final_paulo

- Alumno: Paulo Cesar Andrade Candiotti
- e-mail personal: pauloa699@gmail.com
- e-mail uni: paulo.andrade.c@uni.pe

## Project Name: Credit Risk Classification usando Random Forest

## 1. Descripción General
El objetivo principal es implementar de manera integral 
el ciclo de vida de un modelo de Machine Learning, desde la
definición del problema hasta el despliegue y serving del modelo.

El proyecto sigue buenas prácticas de MLOps, incluyendo: 
- Control de versiones con Github.
- Estructura modular del proyecto.
- Registro de experimentos con MLflow.
- Registro y versionado de modelos.
- Serving del modelo mediante MLflow.

------------------------------------------------------------------------

## 2. Definición del Problema

### Caso de Uso

Se desarrolla un modelo de **clasificación de riesgo crediticio**
utilizando el dataset de créditos alemanes (german_credit_data.csv)

### Objetivo

Predecir si un cliente representa un riesgo crediticio alto o bajo en
función de variables demográficas (edad, genero, vivienda) y 
financieras (cuentas de ahorro, monto del crédito, duracion del préstamo).

### Restricciones

### Beneficios

-   Mejor evaluación del riesgo.
-   Reducción de pérdidas por incumplimiento.


### Resultados esperados

------------------------------------------------------------------------

## 3. Dataset

El dataset incluye variables como:

-   Age: edad del cliente
-   Sex: genero del cliente
-   Job: clasificacion del nivel laboral
-   Housing: tipo de vivienda
-   Saving accounts: saldo en cuenta de ahorros
-   Checking account: saldo en cuenta corriente
-   Credit amount: monto del préstamo
-   Duration: plaza del crédito
-   Purpose: motivo del crédito
-   Risk: Variable objetivo (buen o mal pagador)

El dataset original fue almacenado en:

    data/raw/

Luego de la preparación

Describir preparacion

 y codificación (One-Hot Encoding), el dataset
final de entrenamiento y testeo fue guardado en:

    data/training/

------------------------------------------------------------------------

## 4. Estructura del Proyecto

    proyecto_mlops_final_paulo/
    │
    ├── data/
    │   ├── raw/
    │   └── training/
    │
    ├── experiments/
    ├── models/
    │
    ├── src/
    │   ├── 01_data_preparation.py
    │   ├── 02_train.py
    │   ├── 02_train_mlflow.py
    │   ├── 03_serving.py
    │   └── 04_predict.py
    │   └── 04_predict_mlflow.py
    │
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

## 5. Experimentación en ML

Durante la fase de preparacion se realizaron:

-   Análisis exploratorio de datos (EDA)
-   Transformaciones y codificación de variables

Durante la fase de experimentación se realizaron:

-   Entrenamiento de modelos: Logistic Regression y Random Forest
-   Evaluación de modelos basado en la curva ROC-AUC.


------------------------------------------------------------------------

## 6. Actividades de Desarrollo ML

### Preparación de Datos

Una vez guardado el dataset "data/raw/german_credit_data.csv"
se siguio las siguientes actividades:

-   Cargar el dataset
-   Revision de nulos
-   Convertir el target en 1: buen riesgo, 0:mal riesgo
-   Codificación One-Hot con dummies
-   Separación del dataset en Train/Test
-   Guardado del dataset procesado:
    "data/training/X_train.csv"
    "data/training/X_test.csv"
    "data/training/y_train.csv"
    "data/training/y_test.csv"

### Experimentacion

En esta etapa se experimentó con el modelo Logistic Regression y Random Forest,
obtuvimos las metricas:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Todas los resultados comparativos fueron guardados en "experiments/results.csv"

Después de realizar experimentación en la carpeta experiments, seleccioné Random Forest como modelo campeón basado en ROC-AUC y lo implementé como modelo de producción en src/02_train.py


### Entrenamiento

En esta actividad se usó el modelo ganador en la etapa de experimentacion, que fue el Random forest.
El entrenamiento fue implementado en src/02_train.py
Se guardó el modelo serializado, listo para el API, en la carpeta "models/credit_risk_model.pkl"
Tambien se guardaron las columnas del modelo: "models/model_columns.pkl". Con el fin de asegurar que después de aplicar dummies, el input tenga exactamente la misma estructura que el entrenamiento.


### Entrenamiento con MLFLOW:
El entrenamiento fue implementado en src/02_train_mlflow.py
Se integró MLflow para realizar el seguimiento de experimentos, registrar parámetros, métricas y artefactos, y gestionar versiones del modelo mediante el Model Registry

Registro del modelo:
-   Registrado en MLflow
-   Versionado en Model Registry
-   Marcado con alias "production"


## Despliegue y Serving

En esta actividad se implementó un API REST utilizando Flask para servir el modelo entrenado (models/credit_risk_model.pkl):

El API sigue las siguientes acciones:
- Cargar el modelo serializado de produccion
- Recibe los datos de entrada en formato JSON
- Convertir el input en dataframe.
- Aplicar dummies para replicar el preprocesamiento realizado en entrenamiento.
- Agrega columnas faltantes y la ordena las features utilizado en el entrenamiento.
- Genera la prediccion y probabilidad.



## Inferencia del Modelo (predict)

Se desarrolló un script de la carpeta: src/04_predict.py que consume la API del modelo mediante solicitudes HTTP POST utilizando. Este script valida el correcto funcionamiento del servicio de inferencia simulando con data de entrada y retorna la predicción junto con su probabilidad.


## Inferencia del Modelo (predict) con MLFLOW:
Para el despliegue del modelo se utilizó MLflow, aprovechando su Model Registry y su capacidad de serving como API REST. 
Después del entrenamiento, el modelo fue registrado en el Model Registry con el nombre:
Credit_Risk_Model. MLflow generó versiones automáticas del modelo y se asignó el alias:
@production

En el script ./src/04_predict_mlflow.py se simulan datos de entrada y se envían al endpoint del modelo, validando así el correcto funcionamiento del proceso de inferencia.



## 7. Conclusión

Este proyecto implementa de manera integral el ciclo de vida de Machine
Learning bajo prácticas de MLOps, cubriendo desde la definición del
problema hasta el despliegue del modelo en producción.

Se demuestra la integración de herramientas modernas como MLflow para
garantizar trazabilidad, reproducibilidad y gobernanza del modelo.

------------------------------------------------------------------------
