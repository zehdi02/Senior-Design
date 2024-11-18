import requests

url = "http://127.0.0.1:8000/predict"

img_fp = "../sample_data/manga_page.png"

with open(img_fp, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)

print(response.json())

