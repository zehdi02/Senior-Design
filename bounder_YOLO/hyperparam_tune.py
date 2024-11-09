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
    }

    result_grid = model.tune(
        data="C:\\Users\\mehed\\Documents\\School\\MangaVision\\bounder_YOLO\\manga109_tune.yaml",
        epochs=8,
        patience=4,
        batch=-1,
        nbs=64,
        imgsz=512,
        rect=True,
        device='cuda',
        iterations=16,
        space=search_space
    )

    return


def main():
    tune_yolo_manga109()

if __name__ == '__main__':
    main()
