Requirements:
- For training, make sure to have 'dataset' folder containing 'test', 'train', 'val' folders and 'data.yaml'.
- Each folder containing 'images' and 'labels' folder.
- The 'labels' folder contains .txt file annotations in YOLO format.

1. In this folder, run `uvicorn main:app --reload`.
2. To train, run sample code `endpoint_train.py`.
3. To predict, open `index.html`. Upload manga page and click on 'predict' button. Or, run sample code `endpoint_predict.py`.
4. To evaluate, run sample code `endpoint_evaluate.py`