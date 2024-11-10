import yaml
from ultralytics import YOLO


# Function to evaluate the model and return performance metrics per class
def evaluate_model(results):
    # yolo models return precision, recall, val_loss, mAP50 and mAP50-95 as well as
    metrics = results.results_dict
    precision = metrics['metrics/precision(B)']
    recall = metrics['metrics/recall(B)']
    mAP50 = metrics['metrics/mAP50(B)']
    mAP50_95 = metrics['metrics/mAP50_95(B)']
    fitness = metrics['fitness']


# Function to update class weights based on the error of each class
def adjust_class_weights(class_metrics, current_weights):
    error_threshold = 0.5  # Example threshold to adjust class weights

    for i, metrics in class_metrics.items():
        precision, recall = metrics['precision'], metrics['recall']

        # Adjust weights based on precision or recall
        if precision < error_threshold or recall < error_threshold:
            current_weights[i] = min(current_weights[i] + 0.5, 4.0)  # Increment weight, max 4.0
        else:
            current_weights[i] = max(current_weights[i] - 0.5, 1.0)  # Decrement weight, min 1.0

    return current_weights


def train():

    # Training loop
    for i in range(10):  # Run for 10 iterations
        # Load previous model weights
        if i == 0:
            model = YOLO("yolov8n.pt")
        elif i == 1:
            model = YOLO(f"runs/detect/train/weights/best.pt").to('cuda') if i > 0 else YOLO(
                "yolov8n.pt").to('cuda')
        else:
            model = YOLO(f"runs/detect/train{i}/weights/best.pt").to('cuda') if i > 0 else YOLO(
                "yolov8n.pt").to('cuda')

        # Load the current YAML configuration file
        with open('yolov8.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Adjust class weights in the YAML file based on the model's performance
        if i > 0:  # Skip adjustment for the first loop
            class_metrics = evaluate_model(model, "path/to/validation_data.yaml")
            class_weights = adjust_class_weights(class_metrics)
            config['class_weights'] = class_weights  # Update the class weights in the config

            # Save the adjusted configuration back to the YAML file
            with open('yolov8.yaml', 'w') as file:
                yaml.dump(config, file)

        # for first training using the pretrained model without any training, use tune to optimize hyperparams for the remaining

        # Train the model using the adjusted class weights
        results = model.train(
            data='manga109.yaml',
            epochs=10,
            patience=5,
            batch=12,
            nbs=64,
            imgsz=1024,
            dropout=.05,
            box=9,
            cls=.25,
            dfl=2,
            close_mosiac=0,
            amp=True,
            rect=True,
            augment=True,
            val=True,
            save=True,
            plots=True,
            verbose=True,
            device='cuda'
        )

        # for training after 5th loop include param freeze = True

        # save the model after each iteration
        model.save(f"runs/detect/train{i + 1}/weights/best.pt")

        # evaluate the model after training and print the results
        results = model.val(data="path/to/validation_data.yaml")
        print(f"Training loop {i + 1} results:", results.metrics)


