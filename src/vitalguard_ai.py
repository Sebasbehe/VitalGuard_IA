
# IMPORTAR LIBRERÍAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


print("\n===== CARGANDO DATASET =====\n")

df = pd.read_csv("data/heart.csv")

print(df.head())


# INFORMACIÓN GENERAL

print("\n===== INFORMACIÓN DEL DATASET =====\n")

print(df.info())

print("\n===== VALORES NULOS =====\n")

print(df.isnull().sum())

# VARIABLES DE ENTRADA Y SALIDA

print("\n===== SEPARANDO VARIABLES =====\n")

# Variable objetivo
y = df["DEATH_EVENT"]

# Variables predictoras
X = df.drop("DEATH_EVENT", axis=1)

print("Variables de entrada:")
print(X.columns)

print("\nVariable objetivo:")
print(y.name)

# DIVISIÓN TRAIN / TEST

print("\n===== DIVIDIENDO DATASET =====\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Datos entrenamiento: {X_train.shape}")
print(f"Datos prueba: {X_test.shape}")

# ENTRENAMIENTO DEL MODELO

print("\n===== ENTRENANDO MODELO RANDOM FOREST =====\n")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("Modelo entrenado correctamente.")


# PREDICCIONES

print("\n===== REALIZANDO PREDICCIONES =====\n")

y_pred = model.predict(X_test)

print("Predicciones completadas.")

# MÉTRICAS DEL MODELO

print("\n===== EVALUACIÓN DEL MODELO =====\n")

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy del modelo: {accuracy:.4f}")

print("\n===== REPORTE DE CLASIFICACIÓN =====\n")

print(classification_report(y_test, y_pred))

# MATRIZ DE CONFUSIÓN

print("\n===== MATRIZ DE CONFUSIÓN =====\n")

cm = confusion_matrix(y_test, y_pred)

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Matriz de Confusión - VitalGuard AI")

plt.show()

# IMPORTANCIA DE VARIABLES

print("\n===== IMPORTANCIA DE VARIABLES =====\n")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": importance
})

importance_df = importance_df.sort_values(
    by="Importancia",
    ascending=False
)

print(importance_df)

# GRÁFICA DE IMPORTANCIA

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Variable"],
    importance_df["Importancia"]
)

plt.xlabel("Importancia")
plt.ylabel("Variables")

plt.title("Importancia de Variables - Random Forest")

plt.gca().invert_yaxis()

plt.show()

# PREDICCIÓN INDIVIDUAL

print("\n===== PREDICCIÓN DE PACIENTE =====\n")


sample_patient = [[
    65,      # age
    1,       # anaemia
    250,     # creatinine_phosphokinase
    0,       # diabetes
    35,      # ejection_fraction
    1,       # high_blood_pressure
    250000,  # platelets
    1.3,     # serum_creatinine
    137,     # serum_sodium
    1,       # sex
    0,       # smoking
    120      # time
]]

prediction = model.predict(sample_patient)

probability = model.predict_proba(sample_patient)


# RESULTADO FINAL

if prediction[0] == 1:
    risk = "ALTO RIESGO"
else:
    risk = "BAJO RIESGO"

print(f"\nClasificación del paciente: {risk}")

print("\nProbabilidades:")

print(f"Bajo riesgo: {probability[0][0] * 100:.2f}%")
print(f"Alto riesgo: {probability[0][1] * 100:.2f}%")

print("\n===== VITAL-GUARD AI FINALIZADO =====\n")