import os
import cv2
import xml.etree.ElementTree as ET
from dataset_creator import split_image

# Define root directory paths for images and annotations
ROOT_DIR = '../Manga109'
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, 'annotations')
OUTPUT_DIR = 'Manga109_YOLO'

# Define class names and mappings for easier annotation handling
CLASSES = ['face', 'body', 'text', 'frame']
ID_CLASS_MAP = {0: 'face', 1: 'body', 2: 'text', 3: 'frame'}
CLASS_ID_MAP = {v: k for k, v in ID_CLASS_MAP.items()}

def get_data_manga109(book_title, page_index, root_dir='../Manga109', classes=CLASSES):
    """
    Load an image and its bounding box annotations from the Manga109 dataset.

    Args:
        book_title (str): The title of the manga book.
        page_index (int): The page index in the book.
        root_dir (str): Path to the root directory of the Manga109 dataset.
        classes (list): List of class types to load annotations for.

    Returns:
        tuple: A tuple containing:
            - image (np.ndarray): The loaded grayscale image.
            - annotations (dict): Dictionary of annotations by class, with each bounding box in xyxy format.
    """
    # Define image and annotation directories
    IMAGES_DIR = os.path.join(root_dir, 'images')
    ANNOTATIONS_DIR = os.path.join(root_dir, 'annotations')

    # Load image based on book title and page index
    image_path = os.path.join(IMAGES_DIR, book_title, f'{int(page_index):03}.jpg')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Locate and parse the XML annotation file for the specified book
    annotation_file = [file for file in os.listdir(ANNOTATIONS_DIR) if book_title in file][0]
    tree = ET.parse(os.path.join(ANNOTATIONS_DIR, annotation_file))
    root = tree.getroot()

    # Find the page's annotations
    page = root.find(f".//page[@index='{page_index}']")
    annotations = {class_type: [] for class_type in classes}

    # Extract bounding box data for each specified class
    for class_type in classes:
        for frame in page.findall(class_type):
            xmin = int(frame.get('xmin'))
            xmax = int(frame.get('xmax'))
            ymin = int(frame.get('ymin'))
            ymax = int(frame.get('ymax'))
            annotations[class_type].append([xmin, ymin, xmax, ymax])

    return image, annotations

def manga109_to_yolo(annotations, width, height):
    """
    Convert bounding box annotations from xyxy to YOLO format (x_center, y_center, width, height).

    Args:
        annotations (dict): Dictionary of annotations by class in xyxy format.
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        dict: Dictionary of annotations by class in YOLO format (normalized xywh).
    """
    yolo_annotations = {class_type: [] for class_type in CLASSES}

    # Convert each bounding box to YOLO format
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
    Save an image and its YOLO-format annotations to disk.

    Args:
        image (np.ndarray): Image to save.
        annotations (dict): Dictionary of annotations by class in YOLO format.
        image_output_path (str): Path to save the image.
        label_file_path (str): Path to save the annotations in a YOLO-compatible text file.
    """
    # Save the image to the specified output path
    cv2.imwrite(image_output_path, image)

    # Write annotations to a text file in YOLO format
    with open(label_file_path, 'w') as file:
        for class_type, class_annotations in annotations.items():
            for annotation in class_annotations:
                file.write(f"{CLASS_ID_MAP[class_type]} {' '.join(map(str, annotation))}\n")

def get_data_yolo(image_path):
    """
    Load an image and its YOLO-format labels.

    Args:
        image_path (str): Path to the image file in the YOLO dataset directory.

    Returns:
        tuple: A tuple containing:
            - image (np.ndarray): The loaded image.
            - labels (dict): Dictionary of labels by class in normalized xywh format.
    """
    labels = {class_name: [] for class_name in ID_CLASS_MAP.values()}

    # Load the image
    image = cv2.imread(image_path)

    # Locate the label file and load annotations
    label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            for line in file:
                class_index, x_center, y_center, width, height = map(float, line.strip().split())
                class_name = ID_CLASS_MAP[int(class_index)]
                labels[class_name].append([x_center, y_center, width, height])

    return image, labels

def show_annotated_img_manga109(image, annotations):
    """
    Display an image with annotated bounding boxes for each specified class in xyxy format.

    Args:
        image (np.ndarray): The image to display.
        annotations (dict): Dictionary of annotations by class in xyxy format.
    """
    # Define colors for each class
    colors = {"face": (255, 0, 0), "body": (0, 255, 0), "text": (0, 0, 255), "frame": (255, 255, 0)}

    # Draw each bounding box on the image
    for class_name, bounding_boxes in annotations.items():
        for bbox in bounding_boxes:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[class_name], 2)
            cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_name], 2)

    # Display the image in a window
    cv2.imshow('Annotated Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_annotated_img_yolo(image, labels):
    """
    Display an image with annotated bounding boxes in YOLO (normalized xywh) format.

    Args:
        image (np.ndarray): The image to display.
        labels (dict): Dictionary of labels by class in normalized xywh format.
    """
    colors = {"face": (255, 0, 0), "body": (0, 255, 0), "text": (0, 0, 255), "frame": (255, 255, 0)}

    # Convert and draw each bounding box from YOLO format to displayable format
    for class_name, bounding_boxes in labels.items():
        for bbox in bounding_boxes:
            x_center, y_center, width, height = bbox
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[class_name], 2)
            cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_name], 2)

    # Display the image in a window
    cv2.imshow('Annotated YOLO Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_predictions(model, image_path, display=False):
    """
    Perform YOLO object detection on an image and optionally display the results.

    Args:
        model: YOLO model for prediction.
        image_path (str): Path to the image for prediction.
        display (bool): Whether to display the predictions on the image.

    Returns:
        dict: Dictionary with class names as keys and lists of bounding boxes in xywh format.
    """
    # Define mappings for class names and indices
    class_name_to_index = {"face": 0, "body": 1, "text": 2, "frame": 3}
    index_to_class_name = {v: k for k, v in class_name_to_index.items()}

    # Predict bounding boxes for the image
    results = model.predict(source=image_path, device='cuda')
    bounding_boxes = {name: [] for name in class_name_to_index.keys()}

    # Extract results and map to class names
    class_indices = results[0].boxes.cls.cpu().numpy()
    boxes_xywh = results[0].boxes.xywh.cpu().numpy()
    image = results[0].orig_img.copy()

    for i, cls_index in enumerate(class_indices):
        class_name = index_to_class_name[int(cls_index)]
        box = boxes_xywh[i]
        bounding_boxes[class_name].append(box)

        if display:
            x_center, y_center, width, height = box
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            colors = {"face": (255, 0, 0), "body": (0, 255, 0), "text": (0, 0, 255), "frame": (255, 255, 0)}
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[class_name], 2)
            cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_name], 2)

    if display:
        cv2.imshow("YOLOv8 Class Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return bounding_boxes


def main():
    book_title = "ARMS"
    page_index = 3

    image, annotations = get_data_manga109(book_title, page_index)
    # show_annotated_img_manga109(image, annotations)

    left_image, left_annotations, right_image, right_annotations = split_image(image, annotations)
    # show_annotated_img_yolo(left_image, left_annotations)
    # show_annotated_img_yolo(right_image, right_annotations)

    # width, height = image.shape[1], image.shape[0]
    # annotations = data_to_yolo(annotations, width, height)
    # show_annotated_img_yolo(image, annotations)


if __name__ == '__main__':
    main()