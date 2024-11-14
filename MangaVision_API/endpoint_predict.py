import requests

url = "http://127.0.0.1:8000/predict"

# example request data
img_fp = "./MangaVision_dataset_sample/test/images/YoumaKourin_029_left.jpg"

with open(img_fp, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)

print(response.json())
