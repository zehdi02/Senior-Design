import os
import cv2
from ultralytics import YOLO
from aggregate_runs import aggregate_run_results


def test(model, image_path):
    # Predict and return the result
    results = model.predict(source=image_path, device='cuda')

    # Load and display the original image with predictions
    annotated_img = results[0].plot()  # Plot predictions on the image

    # Show the annotated image using OpenCV
    window_name = "YOLOv8 Prediction"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 1024)
    cv2.imshow(window_name, annotated_img)
    cv2.waitKey(0)  # Press any key to close
    cv2.destroyAllWindows()


def main():
    model = YOLO("runs/detect/train3/weights/best.pt").to('cuda')

    result_grid = model.train(
        data='manga109.yaml',
        epochs=30,
        patience=5,
        batch=12,
        #nbs=64,
        dropout=.05,
        imgsz=1024,
        augment=True,
        val=True,
        save=True,
        plots=True,
        verbose=True,
        device='cuda'
    )

    # aggregate runs
    aggregate_run_results()

    test(model,"Manga109_YOLO/test/images/AisazuNihaIrarenai_017.jpg")
    test(model, "Manga109_YOLO/test/images/AisazuNihaIrarenai_018_left.jpg")



if __name__ == '__main__':
    main()
