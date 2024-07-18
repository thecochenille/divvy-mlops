import requests

import pandas as pd
import pickle

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#set up tracking server
TRACKING_SERVER_HOST = "34.171.118.161" #external IP reserved in GCP
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

print(f"tracking URI: '{mlflow.get_tracking_uri()}'")


data_file = "../data/processed/202304-usage.parquet"

mlflow.set_experiment("experiment-1")


with mlflow.start_run():
    #load prepared data
    df = pd.read_parquet(data_file)
    mlflow.log_param("data_file", data_file)

    features = df[['station_name', 'hour', 'day_of_week']]
    target = df['net_usage']

    features = pd.get_dummies(features, columns=['station_name', 'day_of_week']) #encoding categorical

    split_params_1 = {"test_size": 0.2, "random_state": 42}
    X_train, X_val_test, y_train, y_val_test = train_test_split(features, target, **split_params_1)

    split_params_2 = {"test_size": 0.5, "random_state": 42} #splits test and val 50%
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, **split_params_2)

    #save test set for streamlit
    with open('../data/test_data/202304-usage-test.pkl', 'wb') as f:
        pickle.dump((X_test, y_test), f)

    params = {"n_estimators": 10, "random_state": 42}
    mlflow.log_params(params)

    rf = RandomForestRegressor(**params).fit(X_train, y_train)
    
    y_pred = rf.predict(X_val)
    mlflow.log_metric("mse", mean_squared_error(y_val, y_pred))

    mlflow.sklearn.log_model(rf, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

mlflow.search_experiments()