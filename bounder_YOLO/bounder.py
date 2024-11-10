import os
import cv2
from ultralytics import YOLO
from utils import aggregate_run_results


def get_predict_boxes(model, image_path, display=False):
    """
    Detect and display bounding boxes for specified classes in an image using a YOLO model.

    Args:
        model: YOLO model for prediction.
        image_path (str): Path to the image file.
        display (bool): Whether to display the image with bounding boxes.

    Returns:
        dict: A dictionary where keys are class names and values are lists of bounding boxes in xywh format.
    """

    # Class label mapping
    class_name_to_index = {"face": 0, "body": 1, "text": 2, "frame": 3}
    index_to_class_name = {v: k for k, v in class_name_to_index.items()}

    classes = list(class_name_to_index.keys())  # Default to all classes
    class_indices_to_display = [class_name_to_index[name] for name in classes]

    # Perform prediction
    results = model.predict(source=image_path, device='cuda')

    # Initialize dictionary to store bounding boxes for each class by name
    bounding_boxes = {name: [] for name in class_name_to_index.keys()}

    # Retrieve class labels and bounding box data
    class_indices = results[0].boxes.cls.cpu().numpy()  # Class labels
    boxes_xywh = results[0].boxes.xywh.cpu().numpy()  # Bounding boxes in xywh format
    image = results[0].orig_img.copy()

    # Loop through detected objects
    for i, cls_index in enumerate(class_indices):
        if cls_index in class_indices_to_display:
            # Get the class name and box
            class_name = index_to_class_name[int(cls_index)]
            box = boxes_xywh[i]
            bounding_boxes[class_name].append(box)

            # Extract the center coordinates, width, and height
            x_center, y_center, width, height = box

            if display:
                # Convert xywh to xyxy format for drawing
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)

                # Define colors for each class
                colors = {"face": (255, 0, 0), "body": (0, 255, 0), "text": (0, 0, 255), "frame": (255, 255, 0)}

                # Draw the box on the image with label
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[class_name], 2)
                cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_name], 2)

    if display:
        # Display the image using OpenCV
        window_name = "YOLOv8 Class Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, image.shape[1], image.shape[0])
        cv2.imshow(window_name, image)
        cv2.waitKey(0)  # Press any key to close
        cv2.destroyAllWindows()

    return bounding_boxes


def main():
    model = YOLO("YOLOv8n.pt").to('cuda')

    # result_grid = model.train(
    #     data='manga109.yaml',
    #     epochs=15,
    #     patience=5,
    #     batch=12,
    #     nbs=64,
    #     imgsz=1024,
    #     dropout=.05,
    #     augment=True,
    #     val=True,
    #     save=True,
    #     plots=True,
    #     verbose=True,
    #     device='cuda'
    # )
    # aggregate_run_results()

    image_path = "Manga109_YOLO/train/images/AisazuNihaIrarenai_012_right.jpg"

    # Get bounding boxes for only "text" class (index 2)
    bounding_boxes = get_predict_boxes(model=model, image_path=image_path, display=True)
    print("Bounding boxes for text:", bounding_boxes['text'])


if __name__ == '__main__':
    main()
