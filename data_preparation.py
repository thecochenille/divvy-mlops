import os
import requests
import zipfile
from tqdm import tqdm

import pandas as pd

import datetime





def download_file(files):
    file= '202304' #change to env variables year and month and path
    path= './data/raw'
    url=f'https://divvy-tripdata.s3.amazonaws.com/{file}-divvy-tripdata.zip'

    resp=requests.get(url, stream=True)
    zip_save_path = f'{path}/{file}.zip'

    os.makedirs(path, exist_ok=True)

    with open(zip_save_path,"wb") as handle:
        for data in tqdm(resp.iter_content(chunk_size=1024),
                        desc=f'{file}',
                        postfix=f"save to {zip_save_path}",
                        total=int(resp.headers["Content-Length"])):
            handle.write(data)

    with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
        zip_ref.extractall(path)
    
    os.remove(zip_save_path)


def load_dataframe(filename):
    
    df= pd.read_csv(filename) #"data/202304-divvy-tripdata.csv"
    return df

def remove_missing_data(df):
    df=df.dropna(subset=['start_station_name','end_station_name'])

def create_hour_weekday(df):
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])

    df['started_hour'] = df['started_at'].dt.hour
    df['started_day_of_week'] = df['started_at'].dt.day_name()

    df['ended_hour'] = df['ended_at'].dt.hour
    df['ended_day_of_week'] = df['ended_at'].dt.day_name()

def transform_df_target(df):
    rentals = df.groupby(['start_station_name', 'started_hour', 'started_day_of_week']).size().reset_index(name='average_rentals')
    returns = df.groupby(['end_station_name', 'ended_hour', 'ended_day_of_week']).size().reset_index(name='average_returns')

    usage_df = pd.merge(rentals, returns, left_on=['start_station_name', 'started_hour', 'started_day_of_week'], right_on=['end_station_name', 'ended_hour', 'ended_day_of_week'], how='outer')
    usage_df['average_rentals'] = usage_df['average_rentals'].fillna(0)
    usage_df['average_returns'] = usage_df['average_returns'].fillna(0)
    usage_df['net_usage'] = usage_df['average_rentals'] - usage_df['average_returns']
    return usage_df


def extract_station_info(df):
  df['station_name'] = df.apply(lambda row: row['end_station_name'] if not pd.isna(row['end_station_name']) else row['start_station_name'], axis=1)
  df['hour'] = df.apply(lambda row: row['ended_hour'] if not pd.isna(row['end_station_name']) else row['started_hour'], axis=1)
  df['day_of_week'] = df.apply(lambda row: row['ended_day_of_week'] if not pd.isna(row['end_station_name']) else row['started_day_of_week'], axis=1)

  return df[['net_usage', 'station_name', 'hour', 'day_of_week']]


def main():



    usage_df_2 = extract_station_info(usage_df.copy())
    usage_df_2.to_parquet("") 