# testing /predict endpoint

import requests

url = "http://127.0.0.1:8000/predict"

img_fp = "C:/Users/Zed/Documents/Code/__Senior Design/Senior-Design/MangaVision_API/MangaVision_dataset_sample/test/images/SyabondamaKieta_027_right.jpg"

with open(img_fp, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)

print(response.json())
