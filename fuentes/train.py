# SCRIPT: train.py (Fase 2 MLOps)
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("datos_limp/dataset_estandarizado.csv")
X = df.drop(columns=['Carga_Calefaccion', 'Carga_Refrigeracion'])
y = df['Carga_Calefaccion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Optimizacion_OPEX_HVAC")

def evaluar(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)

with mlflow.start_run(run_name="RandomForest_GridSearch"):
    rf = RandomForestRegressor(random_state=42)
    grid_rf = GridSearchCV(rf, {'n_estimators': [50, 100], 'max_depth': [5, 10]}, cv=3, scoring='neg_mean_squared_error')
    grid_rf.fit(X_train, y_train)
    mejor_rf = grid_rf.best_estimator_
    mae, rmse, r2 = evaluar(y_test, mejor_rf.predict(X_test))
    mlflow.log_params(grid_rf.best_params_)
    mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})
    mlflow.sklearn.log_model(mejor_rf, "modelo_rf", serialization_format="pickle")

with mlflow.start_run(run_name="XGBoost_GridSearch"):
    xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
    grid_xgb = GridSearchCV(xgb, {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}, cv=3, scoring='neg_mean_squared_error')
    grid_xgb.fit(X_train, y_train)
    mejor_xgb = grid_xgb.best_estimator_
    mae, rmse, r2 = evaluar(y_test, mejor_xgb.predict(X_test))
    mlflow.log_params(grid_xgb.best_params_)
    mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})
    mlflow.xgboost.log_model(mejor_xgb, "modelo_xgb")
