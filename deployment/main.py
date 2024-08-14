import streamlit as st
import mlflow
import mlflow.pyfunc

import pandas as pd
import numpy as np
import pickle
import pyarrow as pa

import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import schedule
import time

from utils.model_checker import check_for_model_changes
from utils.model_checker import get_production_models
from utils.download_from_gcs import download_parquet_df_from_gcs, download_pickle_from_gcs, retrieve_encoder_scale_from_pkl



def check_model():
    tracking_uri = "http://34.171.118.161:5000/"
    model_info = get_production_models(tracking_uri)

    model_name = model_info[0]['model_name']
    experiment_name=model_info[0]['experiment_name']
    experiment_id=model_info[0]['experiment_id']

    models_dir ='models'
    version = check_for_model_changes(model_name, tracking_uri, models_dir)
    
    print(f'this is the model name in production: {model_name}-{version}')

    print('Model Check completed!')
    return model_name, version, experiment_name, experiment_id


# getting models dependencies (optional)
#mlflow.pyfunc.get_model_dependencies(logged_model)
#%pip install -r /var/folders/dd/nb8m5vwd1sz3_s49jh1plcfr0000gn/T/tmpkv8uin21/requirements.txt

data_folder_path = 'data/deployment'
models_folder_path = 'models/deployment'
bucket_name = 'mlops-divvy-experiment-tracking'


@st.cache_data
def read_data(experiment_name, experiment_id): #loading test data to show metric
    test_filename = f'202304-transformed_X_test-{experiment_name}-{experiment_id}.parquet.parquet'
    usage_filename = f'20234-usage.parquet.parquet'

    test_data_df = download_parquet_df_from_gcs(bucket_name, data_folder_path, test_filename)
    data_processed = download_parquet_df_from_gcs(bucket_name, data_folder_path, usage_filename) #this set is to extract features values for user to select
    
    return test_data_df, data_processed

@st.cache_resource
def load_model(model_name, version):
    with open(f'models/{model_name}/model-{version}.pkl', 'rb') as f: 
        loaded_model = pickle.load(f)
    return loaded_model

def load_encoder_scaler(experiment_name, experiment_id):
    encoder_scale_filename = f'encoder_scaler-{experiment_name}-{experiment_id}.pkl'
    loaded_object = download_pickle_from_gcs(bucket_name, models_folder_path, encoder_scale_filename)
    encoder, scaler = retrieve_encoder_scale_from_pkl(loaded_object)
    
    # with open('models/encoder-experiment4.pkl', 'rb') as f: 
    #     encoder = pickle.load(f)

    # with open('models/scaler-experiment4.pkl', 'rb') as f: 
    #     scaler = pickle.load(f)
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

def main():
    schedule.every(10).minutes.do(check_model)
    model_name, version, experiment_name, experiment_id = check_model()

    st.title("Chicago Divvy Bike Availability Prediction App")

    st.write("Hello, world! Here, you can predict for Divvy availability. Select the station, day of the week \
            and time you want to ride and this model will predict if there will be bikes available, \
            not too many or not at all")


    #get datasets with divvy stations from predictions
    test_data_df, data_processed = read_data(experiment_name, experiment_id) #this dataset was encoded but not scaled
    loaded_model = load_model(model_name, version)
    encoder, scaler = load_encoder_scaler(experiment_name, experiment_id)

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
                color = "green"
            else:
                result = "There should NOT be any available bike"
                color = "red"
            st.markdown(f'<div style="padding:10px; background-color:{color};">Predicted Bike Availability: {result} at {station_name} on {day_week} around {hour}:00.</div>', unsafe_allow_html=True)
            #st.success(f"Predicted Bike Availability: {result} at {station_name} on {day_week} around {hour}:00.")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
