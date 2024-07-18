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

# getting models dependencies (optional)
#mlflow.pyfunc.get_model_dependencies(logged_model)
#%pip install -r /var/folders/dd/nb8m5vwd1sz3_s49jh1plcfr0000gn/T/tmpkv8uin21/requirements.txt


#load data
@st.cache_data
def read_test_data():
    with open('../data/test_data/202304-usage-test.pkl', 'rb') as f:
        X_test,  y_test = pickle.load(f)
    return X_test, y_test

data_processed = pd.read_parquet("../data/processed/202304-usage.parquet")

@st.cache_data
def load_model():
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model


#functions



def user_predict(model, station_name, hour, day_week): ## still working on this function
    ### notes for next time: https://stackoverflow.com/questions/62240050/how-to-run-model-on-new-data-that-requires-pd-get-dummies
    ### need to change get_dummy to onehotencoder so the predict can ignore missing data from new categories or missing categories in put

    user_input= pd.DataFrame({'station_name': [station_name],'day_of_week':[day_week],'hour':[hour]})

    #user_input = pd.get_dummies(user_input, columns=['station_name', 'day_of_week']) 
    
    user_prediction = model.predict(user_input)
    print(user_prediction)
    return user_prediction

def prediction_score(model,X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test.values, predictions)
    return mse

   

st.title("Divvy Availability prediction App")

st.write("Hello, world!")


#get dataset with divvy stations from predictions
X_test, y_test = read_test_data()
loaded_model = load_model()


# select station
station_name = st.selectbox(
   "Which Divvy Station are you looking for?",
   (data_processed['station_name'].unique().tolist()),
   index=None,
   placeholder="Select Divvy station",
)

st.write("You selected:", station_name)


day_week = st.selectbox(
   "Which day of the week do you want to ride?",
   (data_processed['day_of_week'].unique().tolist()),
   index=None,
   placeholder="Select day",
)

st.write("You selected:", day_week)

hour = st.selectbox(
   "What time will you ride?",
   (data_processed['hour'].unique().tolist()),
   index=None,
   placeholder="Select hour of the day",
)

st.write("You selected:", hour)


if st.button("Predict availability"):
  user_prediction = user_predict(loaded_model, station_name, hour, day_week)
  st.write("Predicted availability:", user_prediction)

  



