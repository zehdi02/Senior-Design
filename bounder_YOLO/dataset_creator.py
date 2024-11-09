import os
import cv2
import utils
import random
from tqdm import tqdm
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed


def setup_directories():
    """Setup train, val, and test directories for images and labels."""
    for dataset_type in ['train', 'val', 'test']:
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}/images', exist_ok=True)
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}/labels', exist_ok=True)


def add_bbox(annotation_dict, class_type, bbox):
    """Add a bounding box to the annotations dictionary."""
    if class_type not in annotation_dict:
        annotation_dict[class_type] = []
    annotation_dict[class_type].append(bbox)


def split_bbox(bbox, mid, width, height):
    """Split and adjust bounding box coordinates for left and right halves."""
    xmin, ymin, xmax, ymax = bbox
    left_bbox = right_bbox = None
    box_area = (xmax - xmin) * (ymax - ymin)

    left_area = max(0, min(xmax, mid) - max(xmin, 0)) * max(0, ymax - ymin)
    right_area = box_area - left_area

    if xmin <= mid <= xmax:
        if left_area / box_area >= 0.4:
            left_bbox = [xmin, ymin, min(xmax, mid), ymax]
        if right_area / box_area >= 0.4:
            right_bbox = [max(xmin - mid, 0), ymin, xmax - mid, ymax]
    elif xmax < mid:
        left_bbox = bbox
    else:
        right_bbox = [xmin - mid, ymin, xmax - mid, ymax]

    return left_bbox, right_bbox


def split_image(image, annotations):
    """
    Split the image and annotations into left and right halves
    """
    width, height = image.shape[1], image.shape[0]
    left_image, right_image = image[:, :width // 2], image[:, width // 2:]
    mid = width // 2
    left_annotations, right_annotations = {}, {}

    for class_type, class_annotations in annotations.items():
        for bbox in class_annotations:
            left_bbox, right_bbox = split_bbox(bbox, mid, width, height)
            if left_bbox:
                add_bbox(left_annotations, class_type, left_bbox)
            if right_bbox:
                add_bbox(right_annotations, class_type, right_bbox)

    return left_image, left_annotations, right_image, right_annotations


def process_page(book, page):
    """
    Process a page by splitting into left and right halves and saving images and labels
    """
    page_index = page.get('index')
    image_path = os.path.join(utils.IMAGES_DIR, book, f'{int(page_index):03}.jpg')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    annotations = {cls: [[int(frame.get(coord)) for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
                  for frame in page.findall(cls)] for cls in utils.CLASSES}

    # Split image and annotations
    left_image, left_annotations, right_image, right_annotations = split_image(image, annotations)

    # Normalize annotations
    left_annotations = utils.manga109_to_yolo(left_annotations, left_image.shape[1], left_image.shape[0])
    right_annotations = utils.manga109_to_yolo(right_annotations, right_image.shape[1], right_image.shape[0])

    # Randomly assign dataset type and paths
    def save_split(image_part, annotations, side):
        dataset_type = random.choices(['train', 'val', 'test'], weights=[0.9, 0.05, 0.05])[0]
        image_output_path = f'{utils.OUTPUT_DIR}/{dataset_type}/images/{book}_{int(page_index):03}_{side}.jpg'
        label_file_path = f'{utils.OUTPUT_DIR}/{dataset_type}/labels/{book}_{int(page_index):03}_{side}.txt'
        utils.save_image_and_labels(image_part, annotations, image_output_path, label_file_path)

    save_split(left_image, left_annotations, 'left')
    save_split(right_image, right_annotations, 'right')


def create_manga109_dataset_yolo():
    """Create dataset by processing each book and page concurrently."""
    setup_directories()
    annotation_files = os.listdir(utils.ANNOTATIONS_DIR)

    with tqdm(total=len(annotation_files), desc="Books Processed") as book_progress:
        with ThreadPoolExecutor() as book_executor:
            book_futures = []
            for annotation_file in annotation_files:
                tree = ET.parse(os.path.join(utils.ANNOTATIONS_DIR, annotation_file))
                root = tree.getroot()
                book = root.get('title')

                with ThreadPoolExecutor() as page_executor:
                    pages = root.findall('.//page')
                    page_futures = [page_executor.submit(process_page, book, page) for page in pages]
                book_futures.append(book_executor.submit(lambda: [f.result() for f in as_completed(page_futures)]))
                book_progress.update(1)

    print("Dataset creation completed.")


def test_create_dataset():
    """Test dataset creation with the first book"""
    setup_directories()
    annotation_file = os.listdir(utils.ANNOTATIONS_DIR)[0]
    tree = ET.parse(os.path.join(utils.ANNOTATIONS_DIR, annotation_file))
    root = tree.getroot()
    book = root.get('title')
    pages = root.findall('.//page')
    for page in pages:
        process_page(book, page)


def main():
    # uncomment to create sample dataset for first book
    # test_create_dataset()

    # uncomment after creating sample dataset to validate annotations remembering to update the imagea path
    # image_path = "Manga109_YOLO/train/images/AisazuNihaIrarenai_045_left.jpg"
    # image, label = utils.get_data_yolo(image_path)
    # utils.show_annotated_img_yolo(image, label)
    #
    # image_path = "Manga109_YOLO/train/images/AisazuNihaIrarenai_045_right.jpg"
    # image, label = utils.get_data_yolo(image_path)
    # utils.show_annotated_img_yolo(image, label)


    # uncomment to create full dataset
    create_manga109_dataset_yolo()

if __name__ == '__main__':
    main()
