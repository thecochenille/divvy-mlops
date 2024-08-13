import mlflow

from utils.model_checker import check_for_model_changes
from utils.model_checker import get_production_models


def main():
    tracking_uri = "http://34.171.118.161:5000/"
    model_name = get_production_models(tracking_uri)
    models_dir ='models'
    print(f'this is the model name in production: {model_name}')
    check_for_model_changes(model_name, tracking_uri, models_dir)

if __name__ == "__main__":
  main()