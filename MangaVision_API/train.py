from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    model = YOLO("yolov8n_MangaVision.pt").to('cuda')

    result_grid = model.train(
        data='MangaVision_dataset_sample/data.yaml',
        epochs=1,
        batch=-1,
        imgsz=640,
        save=True,
        device='cuda',
        verbose=True,
        amp=False,
        dropout=.05,
        val=True,
        plots=True
    )

if __name__ == '__main__':
    main()
