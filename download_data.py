import os
import requests
import zipfile
from tqdm import tqdm

import pandas as pd



def download_file(year, month):
    file= f'{year}{month}' #change to env variables year and month and path
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

    return {
      "path": f'{path}/{year}{month}-divvy-tripdata.csv',
      "year": year,
      "month": month
    }


#def upload_google_storage(data):
#working on adding def to upload data to google storage



def main():
    while True:
        year = input("Monthly Divvy data is available from April 2020. Please enter the year (YYYY) you want data from:")
        if len(year) != 4 or not year.isdigit():
            print("This is not a valid year, please enter a 4 digit year like 2022")
            continue

        month = input("Please enter a month (MM) you want data from:")
        if len(month) != 2 or not month.isdigit():
            print("This is not a valid month, please enter a 2 digit year like 01 for January")
            continue

        result = download_file(year, month)
        
        print(f'Downloaded file is available at {result["path"]}')

        break

if __name__ == "__main__":
    main()




