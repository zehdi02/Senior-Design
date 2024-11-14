import requests
import json

url = 'http://127.0.0.1:8000/train_model'

# example request data
data = {
    'data': 'MangaVision_dataset_sample/data.yaml',
    'epochs': 1,
    'batch': -1,
    'imgsz': 640,
    'save': True,
    'device': 'cuda',
    'verbose': True,
    'amp': False,
    'dropout': .05,
    'val': True,
    'plots': True
}

response = requests.post(url, json=data)

print(response.json())
