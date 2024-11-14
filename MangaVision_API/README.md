Requirements:
- For training, make sure to have 'dataset' folder containing 'test', 'train', 'val' folders and 'data.yaml'.
- Each folder containing 'images' and 'labels' folder.
- The 'labels' folder contains .txt file annotations in YOLO format.

1. In this folder, run `uvicorn main:app --reload`.
2. To train, run sample code `sample_train.py`.
> or, `curl -X POST "http://127.0.0.1:8000/train_model" -H "Content-Type: application/json" -d "@sample_data/train_data.json"`
3. To predict, run sample code `sample_predict.py`
> or,  open `index.html`. Upload a manga page image and click on the 'Predict' button. \
> or, `curl -X POST "http://127.0.0.1:8000/predict" -F "file=@sample_data/manga_page.png"`
4. To evaluate, run sample code `sample_evaluate.py`
> or, `curl -X POST "http://127.0.0.1:8000/evaluate"`
5. To get model status, run `curl http://localhost:8000`

