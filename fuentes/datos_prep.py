# SCRIPT: datos_prep.py (Fase 1 MLOps - Facility Management BBVA)
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
df = pd.read_excel(url)
df.columns = ['Compacidad_Relativa', 'Area_Superficie', 'Area_Pared', 'Area_Techo',
              'Altura_Total', 'Orientacion', 'Area_Acristalamiento',
              'Distribucion_Acristalamiento', 'Carga_Calefaccion', 'Carga_Refrigeracion']
df = df.drop_duplicates()
scaler = StandardScaler()
X = df.drop(columns=['Carga_Calefaccion', 'Carga_Refrigeracion'])
Y = df[['Carga_Calefaccion', 'Carga_Refrigeracion']]
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
df_limpio = pd.concat([X_scaled, Y.reset_index(drop=True)], axis=1)

os.makedirs("datos_limp", exist_ok=True)
os.makedirs("datos_ini", exist_ok=True)
df.to_csv("datos_ini/dataset_original.csv", index=False)
df_limpio.to_csv("datos_limp/dataset_estandarizado.csv", index=False)
