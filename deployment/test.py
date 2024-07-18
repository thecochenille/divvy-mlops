import requests


usage = {
    "net_usage": 4.0,
    "station_name": "2112 W Peterson Ave",
    "hour": 6.0,
	"day_of_week": "Sunday"
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())
