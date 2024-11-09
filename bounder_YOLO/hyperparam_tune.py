import os
import yaml
import shutil
from ultralytics import YOLO


def tune_yolo_manga109():
    model = YOLO("yolov8n.pt")

    search_space = {
        'lr0': (1e-5, 1e-1),
        'lrf': (0.01, 0.5),
        'momentum': (0.6, 0.98),
        'weight_decay': (0.0, 0.001),
        'warmup_epochs': (0.0, 3.0),
        'warmup_momentum': (0.0, 0.95),
        'box': (0.02, 2),
        'cls': (0.2, 4.0),
        'label_smoothing': (0.0, 0.1),

        'hsv_v': (0.0, 1.0),
        'hsv_h': (0.0, 1.0),
        'hsv_s': (0.0, 1.0),
        'degrees': (0.0, 180.0),
        'translate': (0.0, 0.9),
        'scale': (0.0, 0.9),
        'shear': (0.0, 10.0),
        'perspective': (0.0, 0.1),
        'flipud': (0.0, 1.0),
        'fliplr': (0.0, 1.0)
    }

    result_grid = model.tune(
        data="C:\\Users\\mehed\\Documents\\School\\MangaVision\\bounder_YOLO\\manga109_tune.yaml",
        epochs=8,
        fraction=0.5, # inc speed
        patience=4,
        batch=16,
        nbs=64,
        imgsz=256, # inc speed
        device='cuda',
        iterations=32,
        space=search_space
    )

    print(result_grid)

    best_hyperparameters_yaml_path = "runs/detect/tune/best_hyperparameters.yaml"
    with open(best_hyperparameters_yaml_path, 'r') as file:
        best_hyperparameters = yaml.safe_load(file)

    # copy the best hyperparameters to the bounder_YOLO folder
    best_hyperparameters_path = "best_hyperparameters.yaml"
    with open(best_hyperparameters_path, 'w') as file:
        yaml.dump(best_hyperparameters, file)

    os.rename("runs/detect/tune/", "runs/detect/best/")     # Rename tune folder to best
    os.rename("runs/", "tune/")     # Rename runs folder to tune
    detect_path = "tune/detect/"
    for item in os.listdir(detect_path):     # Move all folders inside detect up one level
        shutil.move(os.path.join(detect_path, item), "tune/")
    os.rmdir(detect_path)

    return


def main():
    tune_yolo_manga109()

if __name__ == '__main__':
    main()
