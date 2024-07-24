import streamlit as st
import mlflow
import mlflow.pyfunc

import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



#parameters for model loading from mflow
TRACKING_URL = "http://34.68.82.207:5000"
mlflow.set_tracking_uri(TRACKING_URL)
logged_model = "models:/randomforest-scaled/Production"

# getting models dependencies (optional)
#mlflow.pyfunc.get_model_dependencies(logged_model)
#%pip install -r /var/folders/dd/nb8m5vwd1sz3_s49jh1plcfr0000gn/T/tmpkv8uin21/requirements.txt


#load data
@st.cache_data
def read_data(): #loading test data to show metric
    with open('../data/test_data/202304-usage-experiment2.pkl', 'rb') as f:
        X_train, y_train, X_test, y_test = pickle.load(f)
    return X_train, y_train, X_test, y_test

data_processed = pd.read_parquet("../data/processed/202304-usage.parquet")


#functions
@st.cache_data
def load_model():
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model

def load_encoder_scaler():
    with open('../experiment_tracking/models/encoder-experiment4.pkl', 'rb') as f: 
        encoder = pickle.load(f)

    with open('../experiment_tracking/models/scaler-experiment4.pkl', 'rb') as f: 
        scaler = pickle.load(f)
    
    return encoder, scaler

def bin_predictions(predictions):  ### still working on this

  q1 = np.quantile(predictions, 0.25)
  q3 = np.quantile(predictions, 0.75)

  bins = np.digitize(predictions, [q1, q3])
  bins[bins == 2] = 1  
  bins[bins == 3] = 2

  return bins


def user_predict(model, scaler, encoder, station_name, hour, day_week): ## still working on this function
    ### notes for next time: https://stackoverflow.com/questions/62240050/how-to-run-model-on-new-data-that-requires-pd-get-dummies
    
    user_input= pd.DataFrame({'station_name': [station_name],'day_of_week':[day_week],'hour':[hour]})
    num_features = ['hour']
    cat_features = ['station_name', 'day_of_week']

    #scale num
    scaled_input = pd.DataFrame(scaler.transform(user_input[num_features]), columns=['hour_scaled'])

    #encode cat
    encoded_input = pd.DataFrame(encoder.transform(user_input[cat_features]).toarray(), columns = encoder.get_feature_names_out(cat_features))
    
    user_input = pd.concat([scaled_input,encoded_input], axis=1)
    
    user_prediction = model.predict(user_input)
    binned_prediction = bin_predictions(user_prediction) ## still working on this

    bin_labels = {0: "High Usage", 1: "Some Bikes Available", 2: "Many Bikes Available"}

    return bin_labels[binned_prediction]  ## this is not working



def prediction_score(model,X_test, y_test):  #this is to generate the model score from test set
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test.values, predictions)
    return mse

   

st.title("Divvy Availability prediction App")

st.write("Hello, world!")


#get dataset with divvy stations from predictions
X_train, y_train, X_test, y_test = read_data()
loaded_model = load_model()
encoder, scaler = load_encoder_scaler()


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
  binned_result = user_predict(loaded_model, scaler, encoder, station_name, hour, day_week)
  st.success(f"Predicted Bike Availability: {binned_result}")
  



