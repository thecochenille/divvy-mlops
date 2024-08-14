import pickle
import pyarrow as pa
import pandas as pd
from google.cloud import storage

import mlflow
from mlflow.tracking import MlflowClient


def download_parquet_df_from_gcs(bucket_name, folder_path, file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blob = bucket.blob(f"{folder_path}/{file_name}")
    #data = blob.download_as_string()
    blob.download_to_filename(f'data/{file_name}')

    dataframe = pd.read_parquet(f'data/{file_name}')
    #dataframe = parquet.read_table(io.BytesIO(data)).to_pandas()

    return dataframe


def download_pickle_from_gcs(bucket_name, folder_path, file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{folder_path}/{file_name}.pkl")

    data = blob.download_as_string()
    loaded_object = pickle.loads(data)

    return loaded_object

def retrieve_encoder_scale_from_pkl(loaded_object):
    encoder, scaler = loaded_object
    return encoder, scaler
  
