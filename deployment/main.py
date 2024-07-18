import streamlit as st
import mlflow
import mlflow.pyfunc

import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



#parameters for model loading from mflow
TRACKING_URL = "http://34.171.118.161:5000"
mlflow.set_tracking_uri(TRACKING_URL)
logged_model = "models:/randomforest/Staging"


#load data
with open('../data/test_data/202304-usage-test.pkl', 'rb') as f:
    X_test,  y_test = pickle.load(f)



# getting models dependencies (optional)
#mlflow.pyfunc.get_model_dependencies(logged_model)
#%pip install -r /var/folders/dd/nb8m5vwd1sz3_s49jh1plcfr0000gn/T/tmpkv8uin21/requirements.txt

#load the model
loaded_model = mlflow.pyfunc.load_model(logged_model)


#predict and score
predictions = loaded_model.predict(X_test)
mse = mean_squared_error(y_test.values, predictions)


st.title("My Streamlit App")
st.write("Hello, world!")