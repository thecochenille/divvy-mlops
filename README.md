# MLOps Zoomcamp (Data Talks Club) project
# Predict Divvy bike availability
# Update Jul 18 2024 Currently working on this repository - NOT IN A WORKING STATE NOW

![alt text](https://github.com/thecochenille/divvy-mlops/blob/b4f6c242447d59e3711331b514407471744026d0/images/DIVVY_Bikes_16833634748.jpg)

## Problem Statement
I am a big fan of bike rentals since I first used the Velib in Paris. Now that I live in Chicago, I regularly use Divvy bikes. 
But sometimes, I can be late because I do not plan that at a given time, there would not be any available bikes. 

Can we predict bike availability at Divvy stations on a particular day and hour of the week?

Divvy has provided monthly records of bike usage since April 2020, so I decided to leverage past usage to predict at a given hour of the day, at a given station, how many bikes will be available. In this case, I use a simple random forest model on an engineered target value, which is the net bike usage per hour per station.

For the scope of the MLOps Zoomcamp certification, I am developing an MLOps pipeline that will allow ingestion of data from the website, data preparation for model training, monitoring, and the model to be deployed in a web app. The app allows a user to enter the name of the station and what time they want to use a Divvy bike, and the ML model will output a prediction of high or low availability.

## Starting Model

To implement the MLOps system, I built a basic model where I first engineered bike net usage by station by hour of each weekday as follows:

```count(rentals) - count(returns)```


NB: This value is really simplified and does not account for initial bike numbers in time. It could be improved by accounting for number of bikes per station, but I will focus on the MLOps side for now

Features: hour, day of the week, station

Metric: MSE



## Cloud set up
I following the steps from [kargarisaac.github.io blog post on setting up CGP for the mlops course](https://kargarisaac.github.io/blog/mlops/data%20engineering/2022/06/15/MLFlow-on-GCP.html#Virtual-Machine-as-The-Tracking-Server)

### Experiment tracker setup
Setting up new project on CGP using Terminal

```
gcloud config set project <projectID>
```

### Setting up firewall

```
gcloud compute firewall-rules create mlflow-divvy-server \
    --network default \
    --priority 1000 \
    --direction ingress \
    --action allow \
    --target-tags mlflow-divvy-server \
    --source-ranges 0.0.0.0/0 \
    --rules tcp:5000 \
    --enable-logging
```

## Experiment Tracking
### GCP settup
```
export PROJECT_ID=project-id
export PROJECT_NUMBER=number-id
```

in ~/.bashrc add these lines with the project-id and number-id of your own
```
export PROJECT_ID=mlops-zoomcamp-divvy
export PROJECT_NUMBER=368983834138
```

```
source ~/.bashrc
```


```
gcloud compute instances create mlflow-divvy-server \
    --project=$PROJECT_ID \
    --zone=us-central1-a \
    --machine-type=e2-standard-2 \
    --network-interface=network-tier=PREMIUM,subnet=default \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=$PROJECT_NUMBER-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=mlflow-divvy-tracking-server \
    --create-disk=auto-delete=yes,boot=yes,device-name=mlflow-divvy-tracking-server,image=projects/ubuntu-os-cloud/global/images/ubuntu-2004-focal-v20220610,mode=rw,size=10,type=projects/$PROJECT_ID/zones/us-central1-a/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --reservation-affinity=any
```

Create PostgreSQL instance in GCP console

Access the DB from SSH terminal
```
psql -h CLOUD_SQL_PRIVATE_IP_ADDRESS -U USERNAME DATABASENAME
```


## Workflow orchestration

## Model deployment

Model was deployed on Streamlit




## Model monitoring




===========



## Scripts
Running scripts independently.
The scripts were prepared to be ran in this order


`download_data.py`: script to download monthly datasets. The user running the script needs to specify the year and month (between April 2020 to today). The downloaded data is unzipped and saved into a raw data folder. 

`data_preparation.py`: script loads data from the raw data folder which is cleaned and prepared for ML training or prediction. The user needs to specify year and month as in `download_data.py` and the new dataset generated from this script is daved in processed data folder.

=========
# Credits

- Divvy dataset were download from the Divvy website: https://divvybikes.com/system-data
=======

## Evaluation Criteria/Project tracker

* Problem description
    * 0 points: The problem is not described
    * 1 point: The problem is described but shortly or not clearly 
    * 2 points: The problem is well described and it's clear what the problem the project solves
* Cloud
    * 0 points: Cloud is not used, things run only locally
    * 2 points: The project is developed on the cloud OR uses localstack (or similar tool) OR the project is deployed to Kubernetes or similar container management platforms
    * 4 points: The project is developed on the cloud and IaC tools are used for provisioning the infrastructure
* Experiment tracking and model registry
    * 0 points: No experiment tracking or model registry
    * 2 points: Experiments are tracked or models are registered in the registry
    * 4 points: Both experiment tracking and model registry are used
* Workflow orchestration
    * 0 points: No workflow orchestration
    * 2 points: Basic workflow orchestration
    * 4 points: Fully deployed workflow 
* Model deployment
    * 0 points: Model is not deployed
    * 2 points: Model is deployed but only locally
    * 4 points: The model deployment code is containerized and could be deployed to cloud or special tools for model deployment are used
* Model monitoring
    * 0 points: No model monitoring
    * 2 points: Basic model monitoring that calculates and reports metrics
    * 4 points: Comprehensive model monitoring that sends alerts or runs a conditional workflow (e.g. retraining, generating debugging dashboard, switching to a different model) if the defined metrics threshold is violated
* Reproducibility
    * 0 points: No instructions on how to run the code at all, the data is missing
    * 2 points: Some instructions are there, but they are not complete OR instructions are clear and complete, the code works, but the data is missing
    * 4 points: Instructions are clear, it's easy to run the code, and it works. The versions for all the dependencies are specified.
* Best practices
    * [ ] There are unit tests (1 point)
    * [ ] There is an integration test (1 point)
    * [ ] Linter and/or code formatter are used (1 point)
    * [ ] There's a Makefile (1 point)
    * [ ] There are pre-commit hooks (1 point)
    * [ ] There's a CI/CD pipeline (2 points)




