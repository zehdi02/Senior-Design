import requests

url = "http://127.0.0.1:8000/evaluate"

response = requests.post(url)

print(response.json())
