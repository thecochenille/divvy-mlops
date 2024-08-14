import os
import re
import mlflow
from mlflow.tracking import MlflowClient

import pickle

def extract_version_from_path(file_path):
  match = re.search(r"model-(\d+)\.pkl", file_path)
  if match:
    return match.group(1)
  else:
    return None

def check_for_model_changes(model_name, tracking_uri, models_dir):
  client = MlflowClient(tracking_uri=tracking_uri)
  
  try:
    latest_production_version = client.get_latest_versions(model_name, stages=["Production"])[0]
    current_version = latest_production_version.version
    #print(f'this is the current version in production:{current_version}')
    
    model_dir = os.path.join(models_dir, model_name)  
    print(model_dir)
    model_files = os.listdir(model_dir)
    print(model_files)

    latest_model_file = None
    latest_model_version = None

    for file in model_files:
      version = extract_version_from_path(file)
      
      if version:
        if not latest_model_file or version > latest_model_version:
          latest_model_file = file
          latest_model_version = version


    if latest_model_version != current_version:
        mlflow.set_tracking_uri(tracking_uri)
        #output_path = f'models/model_{current_version}.pkl' #sets the output path as the model version in production
        model_path = os.path.join(models_dir, latest_model_file)
        fetch_model(tracking_uri, model_name, current_version, model_dir)
        os.remove(f'{model_dir}/model-{version}.pkl')
    else:
       print(f'There is no new version of {model_name} in Production. The current version is: {latest_model_version}')
    
    return version
  
  except Exception as e:
    print(f"Error checking model version: {e}")
    

def print_models_info(mv):
    for m in mv:
        print(f"name: {m.name}")
        print(f"latest version: {m.version}")
        print(f"run_id: {m.run_id}")
        print(f"current_stage: {m.current_stage}")



def fetch_model(tracking_uri, model_name, current_version, model_dir):
  #client = MlflowClient(tracking_uri=tracking_uri)

  try:
    #latest_production_version = client.get_latest_versions(model_name, stages=["Production"])[0]

    #print_models_info(client.get_latest_versions(model_name, stages=["Production"]))

    #version = latest_production_version.version
    print(f'The latest model in production:{current_version}')
    #print_models_info(latest_production_version)

    model_uri = f"models:/{model_name}/{current_version}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    #print(f'this is the model uri i need:{model_uri}')
    #production_model = f"models:/{model_name}/Production"
    #loaded_model = mlflow.sklearn.load_model(production_model)
    output_path = f'{model_dir}/model-{current_version}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(loaded_model, f)
    return loaded_model
  
  except Exception as e:
    print(f"Error downloading model: {e}")
    return None


def list_registered_models(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)
    registered_models = client.search_registered_models()

    for model in registered_models:
        latest_versions = client.get_latest_versions(model.name)
        for version in latest_versions:
            print(f"Model Name: {model.name}, Version: {version.version}")

    return model.name


def get_production_models(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)
    registered_models = client.search_registered_models()
    production_models = []
    for model in registered_models:
        latest_versions = client.get_latest_versions(model.name)
        for version in latest_versions:
            if "Production" in version.current_stage:
                production_models.append(model.name)
    return production_models[0]
