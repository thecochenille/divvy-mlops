import pickle
import pyarrow as pa
import pandas as pd
from google.cloud import storage
import io
import os

def save_multiple_dataframes_to_gcs_as_pkl(dataframes, bucket_name, folder_path, file_name):
  """ Saves multiple dataframes (specifically to save the datasets creates by split training and test set function) as a pickle file to Google Cloud Storage.

  Inputs:
    dataframes: A list of pandas DataFrames to be saved.
    bucket_name: The name of the Google Cloud Storage bucket.
    folder_path: The path to the folder within the bucket.
    file_name: The name of the file to be saved.
  """

  with io.BytesIO() as buffer:
    pickle.dump(dataframes, buffer)
    buffer.seek(0)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{folder_path}/{file_name}.pkl")
    blob.upload_from_file(buffer)

def save_dataframe_to_gcs_as_parquet(dataframe, bucket_name, folder_path, file_name):
  """Saves a pandas DataFrame as a parquet file to Google Cloud Storage.

  Inputs:
    dataframe: The pandas DataFrame to be saved.
    bucket_name: The name of the Google Cloud Storage bucket.
    folder_path: The path to the folder within the bucket.
    file_name: The name of the file to be saved.
  """

  parquet_file = f"{file_name}.parquet"
  dataframe.to_parquet(parquet_file)

  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(f"{folder_path}/{parquet_file}")
  blob.upload_from_filename(parquet_file)
 
  os.remove(parquet_file)
