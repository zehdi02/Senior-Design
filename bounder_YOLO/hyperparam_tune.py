import os
import yaml
import shutil
from ultralytics import YOLO

def tune_yolo_manga109():
    """
    Tunes the YOLOv8 model on the Manga109 dataset by performing hyperparameter search.
    Saves the best-performing hyperparameters and reorganizes output directories for clarity.
    """
    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Define hyperparameter search space for model tuning
    search_space = {
        # Model hyperparameters
        'lr0': (1e-5, 1e-1),
        'lrf': (0.01, 0.5),
        'momentum': (0.6, 0.98),
        'weight_decay': (0.0, 0.001),
        'warmup_epochs': (0.0, 3.0),
        'warmup_momentum': (0.0, 0.95),
        'box': (0.02, 2),
        'cls': (0.2, 4.0),
        'label_smoothing': (0.0, 0.1),

        # Augmentation hyperparameters
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

    # Execute model tuning
    result_grid = model.tune(
        data="C:\\Users\\mehed\\Documents\\School\\MangaVision\\bounder_YOLO\\yaml_files\\manga109.yaml",
        epochs=8,
        fraction=0.5,  # Fraction of dataset to speed up tuning
        patience=4,  # Early stopping if no improvement
        batch=16,
        nbs=64,  # Nominal batch size for larger datasets
        imgsz=256,  # Reduced image size for faster processing
        device='cuda',
        iterations=32,  # Number of tuning iterations
        space=search_space
    )
    print("Tuning completed. Result grid:", result_grid)

    # Load and save the best hyperparameters
    best_hyperparameters_yaml_path = "runs/detect/tune/best_hyperparameters.yaml"
    with open(best_hyperparameters_yaml_path, 'r') as file:
        best_hyperparameters = yaml.safe_load(file)

    # Save best hyperparameters to a specified location for easy access
    best_hyperparameters_output_path = "yaml_files/best_hyperparameters.yaml"
    with open(best_hyperparameters_output_path, 'w') as file:
        yaml.dump(best_hyperparameters, file)

    # Rename output directories to improve organization
    os.rename("runs/detect/tune/", "runs/detect/best/")  # Rename tuning output folder to 'best'
    os.rename("runs/", "tune/")  # Rename main 'runs' folder to 'tune'

    # Move all items from the 'tune/detect' directory up one level and clean up
    detect_path = "tune/detect/"
    for item in os.listdir(detect_path):
        shutil.move(os.path.join(detect_path, item), "tune/")
    os.rmdir(detect_path)  # Remove now-empty 'detect' directory

def main():
    """Main function to initiate YOLO model tuning on Manga109 dataset."""
    tune_yolo_manga109()

if __name__ == '__main__':
    main()
