import cv2
from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/train3/weights/best.pt").to('cuda')

    result_grid = model.train(
        data='manga109.yaml',
        epochs=100,
        patience=5,
        batch=8,
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


    img_path = f"Manga109_YOLO/test/images/AisazuNihaIrarenai_017.jpg"
    # Predict and return the result
    results = model.predict(source=img_path, device='cuda')

    # Load and display the original image with predictions
    img = cv2.imread(img_path)
    annotated_img = results[0].plot()  # Plot predictions on the image

    # Show the annotated image using OpenCV
    cv2.imshow("YOLOv8 Prediction", annotated_img)
    cv2.waitKey(0)  # Press any key to close
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
