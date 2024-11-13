Make sure to have Manga109 dataset in this directory which include manga pages and their labels, as well as data.yaml.
1. In this folder, run `uvicorn main:app --reload`.
2. To train and get evaluations, run `endpoint_train.py`.
3. To predict, open `index.html`. Upload manga page and click on 'predict' button. Or, run `endpoint_predict.py`.