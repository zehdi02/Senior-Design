import os
import cv2
import random
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Import tqdm for progress bars


ROOT_DIR = '../Manga109'
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, 'annotations')
OUTPUT_DIR = 'Manga109_YOLO'

def assign_dataset_type():
    random_number = random.random()
    if random_number < 0.90:
        return 'train'
    elif random_number < 0.95:
        return 'val'
    else:
        return 'test'

def process_page(book_title, page_index, image_path, page):
    try:
        # Randomly assign dataset type for left and right images
        left_dataset_type = assign_dataset_type()
        right_dataset_type = assign_dataset_type()

        # Output paths for left and right split images
        left_image_output_path = f'{OUTPUT_DIR}/{left_dataset_type}/images/{book_title}_{int(page_index):03}_left.jpg'
        right_image_output_path = f'{OUTPUT_DIR}/{right_dataset_type}/images/{book_title}_{int(page_index):03}_right.jpg'

        # Load the image and get dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"Image not found: {image_path}")
            return
        width = int(page.get('width'))
        height = int(page.get('height'))

        # Write left and right images
        cv2.imwrite(left_image_output_path, img[:, :width // 2])
        cv2.imwrite(right_image_output_path, img[:, width // 2:])

        # Paths for left and right label files
        left_label_file_path = f'{OUTPUT_DIR}/{left_dataset_type}/labels/{book_title}_{int(page_index):03}_left.txt'
        right_label_file_path = f'{OUTPUT_DIR}/{right_dataset_type}/labels/{book_title}_{int(page_index):03}_right.txt'

        with open(left_label_file_path, 'w') as left_label_file, \
             open(right_label_file_path, 'w') as right_label_file:
            # Process annotations for each object type
            for i, frame_type in enumerate(['face', 'body', 'text', 'frame']):
                for frame in page.findall(frame_type):
                    xmin = int(frame.get('xmin'))
                    xmax = int(frame.get('xmax'))
                    ymin = int(frame.get('ymin'))
                    ymax = int(frame.get('ymax'))

                    # Normalized height and y_center (common for both halves)
                    norm_height = (ymax - ymin) / height
                    norm_y_center = (ymin + ymax) / (2 * height)

                    if xmax <= width // 2:
                        norm_x_center_left = (xmin + xmax) / (2 * (width // 2))
                        norm_width_left = (xmax - xmin) / (width // 2)
                        left_label_file.write(f'{i} {norm_x_center_left} {norm_y_center} {norm_width_left} {norm_height}\n')
                    elif xmin >= width // 2:
                        norm_x_center_right = ((xmin - width // 2) + (xmax - width // 2)) / (2 * (width // 2))
                        norm_width_right = (xmax - xmin) / (width // 2)
                        right_label_file.write(f'{i} {norm_x_center_right} {norm_y_center} {norm_width_right} {norm_height}\n')

    except Exception as e:
        print(f"Error processing page {page_index} of book {book_title}: {e}")

def create_dataset_yolo():
    # Create train, val, test directories if not already present
    for dataset_type in ['train', 'val', 'test']:
        os.makedirs(f'{OUTPUT_DIR}/{dataset_type}/images', exist_ok=True)
        os.makedirs(f'{OUTPUT_DIR}/{dataset_type}/labels', exist_ok=True)

    # Initialize a progress bar for books
    annotation_files = os.listdir(ANNOTATIONS_DIR)
    with tqdm(total=len(annotation_files), desc="Books Processed") as book_progress:
        # Process each annotation file (each book) concurrently
        with ThreadPoolExecutor() as book_executor:
            book_futures = []
            for annotation_file in annotation_files:
                tree = ET.parse(os.path.join(ANNOTATIONS_DIR, annotation_file))
                root = tree.getroot()
                book_title = root.get('title')

                # Process pages within each book concurrently, with a progress bar for pages
                page_futures = []
                with ThreadPoolExecutor() as page_executor:
                    pages = root.findall('.//page')
                    with tqdm(total=len(pages), desc=f"Pages in {book_title}", leave=False) as page_progress:
                        for page in pages:
                            page_index = page.get('index')
                            image_path = os.path.join(IMAGES_DIR, book_title, f'{int(page_index):03}.jpg')
                            future = page_executor.submit(process_page, book_title, page_index, image_path, page)
                            future.add_done_callback(lambda _: page_progress.update(1))  # Update page progress
                            page_futures.append(future)

                # Wait for all page processing to complete for the current book
                book_futures.append(book_executor.submit(lambda: [f.result() for f in as_completed(page_futures)]))
                book_progress.update(1)  # Update book progress

    print("Dataset creation completed.")

def main():
    create_dataset_yolo()

if __name__ == '__main__':
    main()
