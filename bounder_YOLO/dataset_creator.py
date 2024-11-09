import os
import cv2
import utils
import random
from tqdm import tqdm
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed


def split_image(image, annotations):
    """
    Split the image and annotations into left and right halves

    Args:
        image: numpy array of the image
        annotations: dictionary containing annotations for each class

    Returns:
        left_image: numpy array of the left half of the image
        left_annotations: dictionary containing annotations for each class in the left half
        right_image: numpy array of the right half of the image
        right_annotations: dictionary containing annotations for each class in the right half

    """
    # Split the image and annotations into left and right halves returns images and annotations for both halves in manga109 format
    width, height = image.shape[1], image.shape[0]
    left_image, right_image = image[:, :width // 2], image[:, width // 2:]
    mid = width // 2
    left_annotations, right_annotations = {}, {}

    for class_type, class_annotations in annotations.items():
        for bbox in class_annotations:
            xmin, ymin, xmax, ymax = bbox

            # only if bounding box spans both halves
            if xmin <= mid <= xmax:
                # calculate total area of bounding box
                box_area = (xmax - xmin) * (ymax - ymin)
                left_area = max(0, min(xmax, width // 2) - max(xmin, 0)) * max(0, min(ymax, height) - max(ymin, 0))
                right_area = box_area - left_area

                if left_area / box_area >= 0.4:
                    if class_type not in left_annotations:
                        left_annotations[class_type] = []
                    left_annotations[class_type].append(bbox)
                if right_area / box_area >= 0.4:
                    if class_type not in right_annotations:
                        right_annotations[class_type] = []
                    right_annotations[class_type].append(bbox)

            elif xmax < mid:
                if class_type not in left_annotations:
                    left_annotations[class_type] = []
                left_annotations[class_type].append(bbox)
            else:
                if class_type not in right_annotations:
                    right_annotations[class_type] = []
                right_annotations[class_type].append(bbox)

    return left_image, left_annotations, right_image, right_annotations


def process_page(book, page):
    """
    Process a page from the manga109 dataset by splitting it into left and right halves and saving the images and labels

    Args:
        book: name of the book
        page: xml element of the page

    """
    # Get the image path
    page_index = page.get('index')
    image_path = os.path.join(utils.IMAGES_DIR, book, f'{int(page_index):03}.jpg')

    # Read the image and annotations
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    annotations = {}
    for object in utils.CLASSES:
        for frame in page.findall(object):
            xmin = int(frame.get('xmin'))
            xmax = int(frame.get('xmax'))
            ymin = int(frame.get('ymin'))
            ymax = int(frame.get('ymax'))

            if object not in annotations:
                annotations[object] = []
            annotations[object].append([xmin, ymin, xmax, ymax])

    # split the image and annotations into left and right halves
    left_image, left_annotations, right_image, right_annotations = split_image(image, annotations)
    # normalize the annotations
    left_annotations = utils.manga109_to_yolo(left_annotations, int(left_image.shape[1]), int(left_image.shape[0]))
    right_annotations = utils.manga109_to_yolo(right_annotations, int(right_image.shape[1]), int(right_image.shape[0]))

    # Randomly assign dataset type for left and right images
    left_dataset_type = random.choices(['train', 'val', 'test'], weights=[0.9, 0.05, 0.05])[0]
    left_image_output_path = f'{utils.OUTPUT_DIR}/{left_dataset_type}/images/{book}_{int(page_index):03}_left.jpg'
    left_label_file_path = f'{utils.OUTPUT_DIR}/{left_dataset_type}/labels/{book}_{int(page_index):03}_left.txt'

    right_dataset_type = random.choices(['train', 'val', 'test'], weights=[0.9, 0.05, 0.05])[0]
    right_image_output_path = f'{utils.OUTPUT_DIR}/{right_dataset_type}/images/{book}_{int(page_index):03}_right.jpg'
    right_label_file_path = f'{utils.OUTPUT_DIR}/{right_dataset_type}/labels/{book}_{int(page_index):03}_right.txt'

    # Save the left and right images and labels in yolo text format
    utils.save_image_and_labels(left_image, left_annotations, left_image_output_path, left_label_file_path)
    utils.save_image_and_labels(right_image, right_annotations, right_image_output_path, right_label_file_path)

    return


def create_manga109_dataset_yolo():
    """
    Create the manga109 dataset in YOLO format, splitting the images and labels into left and right halves, concurrently for each book and page
    """
    # Create train, val, test directories if not already present
    for dataset_type in ['train', 'val', 'test']:
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}/images', exist_ok=True)
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}/labels', exist_ok=True)

    # Initialize a progress bar for books
    annotation_files = os.listdir(utils.ANNOTATIONS_DIR)
    with tqdm(total=len(annotation_files), desc="Books Processed") as book_progress:
        # Process each annotation file (each book) concurrently
        with ThreadPoolExecutor() as book_executor:
            book_futures = []
            for annotation_file in annotation_files:
                tree = ET.parse(os.path.join(utils.ANNOTATIONS_DIR, annotation_file))
                root = tree.getroot()
                book = root.get('title')

                # Process pages within each book concurrently, with a progress bar for pages
                page_futures = []
                with ThreadPoolExecutor() as page_executor:
                    pages = root.findall('.//page')
                    for page in pages:
                        future = page_executor.submit(process_page, book, page)
                        page_futures.append(future)

                # Wait for all page processing to complete for the current book
                book_futures.append(book_executor.submit(lambda: [f.result() for f in as_completed(page_futures)]))
                book_progress.update(1)  # Update book progress

    print("Dataset creation completed.")

    return


def main():
    create_manga109_dataset_yolo()
    return

if __name__ == '__main__':
    main()
