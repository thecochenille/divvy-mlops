import streamlit as st
import mlflow
import mlflow.pyfunc

import pandas as pd
import numpy as np
import pickle

import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



#ONHOLD USING DOWNLOADED MODEL FIRST - parameters for model loading from mflow / on hold
#TRACKING_URL = "http://34.171.118.161:5000"
#mlflow.set_tracking_uri(TRACKING_URL)
#logged_model = "models:/randomforest-scaled/Production"

# getting models dependencies (optional)
#mlflow.pyfunc.get_model_dependencies(logged_model)
#%pip install -r /var/folders/dd/nb8m5vwd1sz3_s49jh1plcfr0000gn/T/tmpkv8uin21/requirements.txt


#load data
@st.cache_data
def read_data(): #loading test data to show metric
    test_data_df = pd.read_parquet('data/202304-test-transformed.parquet')
    data_processed = pd.read_parquet("data/202304-usage.parquet")
    return test_data_df, data_processed


#functions
@st.cache_resource
def load_model():
    with open('models/model.pkl', 'rb') as f: 
        loaded_model = pickle.load(f)
    #loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model

def load_encoder_scaler():
    with open('models/encoder-experiment4.pkl', 'rb') as f: 
        encoder = pickle.load(f)

    with open('models/scaler-experiment4.pkl', 'rb') as f: 
        scaler = pickle.load(f)
    
    return encoder, scaler

def calculate_bins(predictions):
  q1 = np.quantile(predictions, 0.25)
  q3 = np.quantile(predictions, 0.75)
  return [q1, q3]

def assign_to_bin(prediction, bins):
  if prediction <= bins[0]:
    return 0  # Many bikes available
  elif prediction <= bins[1]:
    return 1  # Some bikes available
  else:
    return 2  # High usage


def user_predict(model, scaler, encoder, station_name, hour, day_week): ## still working on this function
    ### notes for next time: https://stackoverflow.com/questions/62240050/how-to-run-model-on-new-data-that-requires-pd-get-dummies
    
    user_input= pd.DataFrame({'station_name': [station_name],'day_of_week':[day_week],'hour':[hour]})
    num_features = ['hour']
    cat_features = ['station_name', 'day_of_week']

    scaled_input = pd.DataFrame(scaler.transform(user_input[num_features]), columns=['hour_scaled'])
    encoded_input = pd.DataFrame(encoder.transform(user_input[cat_features]).toarray(), columns = encoder.get_feature_names_out(cat_features))
    user_input = pd.concat([scaled_input,encoded_input], axis=1)

    user_prediction = model.predict(user_input)
    return user_prediction 


def prediction_score(model, test_data):  #this is to generate the model score from test set
    X_test = test_data.drop('net_usage', axis=1)
    y_test = test_data['net_usage']

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test.values, predictions)
    return predictions, mse

st.title("Chicago Divvy Bike Availability Prediction App")

st.write("Hello, world! Here, you can predict for Divvy availability. Select the station, day of the week \
         and time you want to ride and this model will predict if there will be bikes available, \
         not too many or not at all")


#get datasets with divvy stations from predictions
test_data_df, data_processed = read_data() #this dataset was encoded but not scaled
loaded_model = load_model()
encoder, scaler = load_encoder_scaler()

# from test predictions
# calculate mse for display

test_predictions, test_mse = prediction_score(loaded_model, test_data_df)

# calculate bins
bins = calculate_bins(test_predictions)

col1, col2 = st.columns(2)
with col1:
    st.header("Model Performance")
    st.write("Mean Squared Error on our test set:", test_mse)

    df = pd.DataFrame({'y_test': test_data_df['net_usage'], 'y_pred': test_predictions})

    fig = px.histogram(df, barmode='overlay', histnorm='probability density')
    st.plotly_chart(fig)


with col2: 
    st.header("Predict Divvy Availability")
    
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

    #hour = st.selectbox(
    #"What time will you ride?",
    #(data_processed['hour'].unique().tolist()),
    #index=None,
    #placeholder="Select hour of the day",
    #)
    #st.write("You selected:", hour)"


    selected_time = st.time_input("What time will you ride?")
    hour = selected_time.hour
    st.write('You selected:', selected_time)



    if st.button("Predict availability"):
        user_prediction = user_predict(loaded_model, scaler, encoder, station_name, hour, day_week)
        if user_prediction < 0: #since usage is rentals-returns , more returns means net_usage < 0 and therefore bikes available
           result = "There should be bikes available"
        else:
           result = "There should NOT be any available bike"
        st.success(f"Predicted Bike Availability: {result} at {station_name} on {day_week} around {hour}:00.")
  

