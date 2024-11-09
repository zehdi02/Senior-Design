import os
import cv2
import xml.etree.ElementTree as ET
from dataset_creator import split_image


ROOT_DIR = '../Manga109'
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, 'annotations')
OUTPUT_DIR = 'Manga109_YOLO'
CLASSES = ['face', 'body', 'text', 'frame']
ID_CLASS_MAP = {0: 'face', 1: 'body', 2: 'text', 3: 'frame'}
CLASS_ID_MAP = {'face': 0, 'body': 1, 'text': 2, 'frame': 3}

def get_data_manga109(book_title, page_index, root_dir='../Manga109', classes=CLASSES):
    """
    Read an image and its corresponding labels from the Manga109 dataset.

    Args:
        book_title (str): Title of the manga book.
        page_index (int): Page index.
        root_dir (str): Root directory of the Manga109 dataset. Defaults to '../Manga109'.

    Returns:
        tuple: A tuple containing the image and dictionary of annotations by class in xyxy format.
    """

    IMAGES_DIR = os.path.join(root_dir, 'images')
    ANNOTATIONS_DIR = os.path.join(root_dir, 'annotations')

    # get image and annotation
    image_path = os.path.join(IMAGES_DIR, book_title, f'{int(page_index):03}.jpg')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    annotation_files = os.listdir(ANNOTATIONS_DIR)
    annotation_file = [file for file in annotation_files if book_title in file][0]
    tree = ET.parse(os.path.join(ANNOTATIONS_DIR, annotation_file))
    root = tree.getroot()
    page = root.find(f".//page[@index='{page_index}']")

    annotations = {class_type: [] for class_type in classes}
    for i, class_type in enumerate(classes):
        for frame in page.findall(class_type):
            xmin = int(frame.get('xmin'))
            xmax = int(frame.get('xmax'))
            ymin = int(frame.get('ymin'))
            ymax = int(frame.get('ymax'))

            annotations[class_type].append([xmin, ymin, xmax, ymax])

    return image, annotations

def manga109_to_yolo(annotations, width, height):
    """
    Convert annotations in xyxy format to YOLO format (x_center, y_center, width, height).

    Args:
        image (np.ndarray): Image to get the shape of the bounding boxes.
        annotations (dict): Dictionary of annotations by class in xyxy format.
        classes (list): List of class names. Defaults to ['face', 'body', 'text', 'frame'].

    Returns:
        dict: Dictionary of annotations by class in YOLO format.
    """
    yolo_annotations = {class_type: [] for class_type in CLASSES}

    for class_type, class_annotations in annotations.items():
        for bbox in class_annotations:
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height
            yolo_annotations[class_type].append([x_center, y_center, bbox_width, bbox_height])

    return yolo_annotations


def save_image_and_labels(image, annotations, image_output_path, label_file_path):
    """
    Save an image and its annotations to disk.

    Args:
        image (np.ndarray): Image to save.
        annotations (dict): Dictionary of annotations by class in yolo normalized xywh format.
        image_output_path (str): Path to save the image.
        label_file_path (str): Path to save the annotations in YOLO format.
    """
    cv2.imwrite(image_output_path, image)
    with open(label_file_path, 'w') as file:
        for class_type, class_annotations in annotations.items():
            for annotation in class_annotations:
                file.write(f"{CLASS_ID_MAP[class_type]} {' '.join(map(str, annotation))}\n")

    return

def get_data_yolo(image_path):
    """
    Read an image and its labels after it has been processed to be YOLO compatible dataset

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing the image and a list of labels.
    """
    class_mapping = {0: 'face', 1: 'body', 2: 'text', 3: 'frame'}
    labels = {class_name: [] for class_name in class_mapping.values()}

    # Read the image
    image = cv2.imread(image_path)

    # Get labels from the image path
    label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
    with open(label_path, 'r') as file:
        for line in file:
            class_index, x_center, y_center, width, height = map(float, line.strip().split())
            class_name = class_mapping[int(class_index)]
            labels[class_name].append([x_center, y_center, width, height])

    return image, labels


def show_annotated_img_manga109(image, annotations):
    """
    Display an image with bounding boxes for specified classes.

    Args:
        image (np.ndarray): Image to display.
        annotations (dict): Dictionary of annotations by class in xyxy format.
    """
    # Define colors for each class
    colors = {"face": (255, 0, 0), "body": (0, 255, 0), "text": (0, 0, 255), "frame": (255, 255, 0)}

    # Draw bounding boxes on the image
    for class_name, bounding_boxes in annotations.items():
        for bbox in bounding_boxes:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[class_name], 2)
            cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_name], 2)

    # Display the image
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def show_annotated_img_yolo(image, labels):
    """
    Display an image with bounding boxes for specified classes.

    Args:
        image (np.ndarray): Image to display.
        labels (dict): Dictionary of labels by class in xywh format.

    """
    # Define colors for each class
    colors = {"face": (255, 0, 0), "body": (0, 255, 0), "text": (0, 0, 255), "frame": (255, 255, 0)}

    # Draw bounding boxes on the image
    for class_name, bounding_boxes in labels.items():
        for bbox in bounding_boxes:
            x_center, y_center, width, height = bbox
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[class_name], 2)
            cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_name], 2)

    # Display the image
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def get_predictions(model, image_path, display=False):
    """
    Detect and display bounding boxes for specified classes in an image using a YOLO model.

    Args:
        model: YOLO model for prediction.
        image_path (str): Path to the image file.
        classes (list, optional): List of class names to display bounding boxes for.
                                             Defaults to None, which displays all classes.

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
    book_title = "ARMS"
    page_index = 3

    image, annotations = get_data_manga109(book_title, page_index)
    # show_annotated_img_manga109(image, annotations)

    left_image, left_annotations, right_image, right_annotations = split_image(image, annotations)
    print(left_annotations)
    show_annotated_img_yolo(left_image, left_annotations)

    # show_annotated_img_yolo(right_image, right_annotations)

    # width, height = image.shape[1], image.shape[0]
    # annotations = data_to_yolo(annotations, width, height)
    # show_annotated_img_yolo(image, annotations)


if __name__ == '__main__':
    main()