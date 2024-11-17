import requests
import json

url = "http://127.0.0.1:8000/evaluate"
# example request data
data = { 'data': 'MangaVision_dataset_sample/data.yaml' }
response = requests.post(url, json=data)
print(response.json())
