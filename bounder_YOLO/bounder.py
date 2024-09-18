from ultralytics import YOLO
import cv2


def main():
    model = YOLO("runs/detect/train/weights/best.pt").to('cuda')

    # result_grid = model.train(
    #     data='manga109.yaml',
    #     epochs=50,
    #     batch=16,
    #     device='cuda',
    #     dropout=.05,
    #     val=True,
    #     save=True
    # )

    img_path = "Manga109_YOLO/test/images/Arisa_063.jpg"
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


"""
"""