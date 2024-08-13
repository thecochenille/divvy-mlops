import os
import requests
import zipfile
from tqdm import tqdm

import pandas as pd

import datetime



def load_dataframe(filename):
    df= pd.read_csv(filename) 
    return df

def remove_missing_data(df):
    df = df.dropna(subset=['start_station_name','end_station_name'])
    return df

def create_hour_weekday(df):
    ''' taking a raw dataframe from Divvy and creates an hour and day of the week feature
    for started_at and ended_at (timestamps)

    input: dataframe
    output: dataframe with 4 additional columns
    '''
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])

    df['started_hour'] = df['started_at'].dt.hour
    df['started_day_of_week'] = df['started_at'].dt.day_name()

    df['ended_hour'] = df['ended_at'].dt.hour
    df['ended_day_of_week'] = df['ended_at'].dt.day_name()

    return df

def transform_df_target(df):
    '''
    '''

    rentals = df.groupby(['start_station_name', 'started_hour', 'started_day_of_week']).size().reset_index(name='average_rentals')
    returns = df.groupby(['end_station_name', 'ended_hour', 'ended_day_of_week']).size().reset_index(name='average_returns')

    usage_df = pd.merge(rentals, returns, left_on=['start_station_name', 'started_hour', 'started_day_of_week'], right_on=['end_station_name', 'ended_hour', 'ended_day_of_week'], how='outer')
    usage_df['average_rentals'] = usage_df['average_rentals'].fillna(0)
    usage_df['average_returns'] = usage_df['average_returns'].fillna(0)
    usage_df['net_usage'] = usage_df['average_rentals'] - usage_df['average_returns']
    
    return usage_df


def extract_station_info(df):
  '''
  '''
  df['station_name'] = df.apply(lambda row: row['end_station_name'] if not pd.isna(row['end_station_name']) else row['start_station_name'], axis=1)
  df['hour'] = df.apply(lambda row: row['ended_hour'] if not pd.isna(row['end_station_name']) else row['started_hour'], axis=1)
  df['day_of_week'] = df.apply(lambda row: row['ended_day_of_week'] if not pd.isna(row['end_station_name']) else row['started_day_of_week'], axis=1)

  return df[['net_usage', 'station_name', 'hour', 'day_of_week']] #this is the final dataset I need


def main():
    while True:
        year = input("Please enter the year (YYYY) you want for data preparation:")
        if len(year) != 4 or not year.isdigit():
            print("This is not a valid year, please enter a 4 digit year like 2022")
            continue

        month = input("Please enter a month (MM) you want data from:")
        if len(month) != 2 or not month.isdigit():
            print("This is not a valid month, please enter a 2 digit year like 01 for January")
            continue

        path_input ='data/raw'   
        path_output ='data/processed'
        filename_input = f'{path_input}/{year}{month}-divvy-tripdata.csv'
        filename_output = f'{path_output}/{year}{month}-usage.parquet'


        df = load_dataframe(filename_input) 
        df = remove_missing_data(df)
        df = create_hour_weekday(df)

        usage_df = transform_df_target(df)

        usage_df_final = extract_station_info(usage_df.copy())

        usage_df_final.to_parquet(filename_output)  

        break

if __name__ == "__main__":
    main()