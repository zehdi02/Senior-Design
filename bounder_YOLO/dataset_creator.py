import os
import cv2
import shutil
import random
import xml.etree.ElementTree as ET

ROOT_DIR = '../Manga109'
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, 'annotations')
DATASET_TYPES = ['train', 'val', 'test']

def create_dataset_yolo():
    IMG_COUNTER = 0

    # Create train, val, test directories
    for dataset_type in DATASET_TYPES:
        os.makedirs(f'Manga109_YOLO/{dataset_type}/images', exist_ok=True)
        os.makedirs(f'Manga109_YOLO/{dataset_type}/labels', exist_ok=True)

    # Loop through all the annotation files
    for annotation_file in os.listdir(ANNOTATIONS_DIR):
        tree = ET.parse(os.path.join(ANNOTATIONS_DIR, annotation_file))
        root = tree.getroot()
        book_title = root.get('title')

        # Loop through pages in the annotation file
        for page in root.findall('.//page'):
            page_index = page.get('index')
            image_path = os.path.join(IMAGES_DIR, book_title, f'{int(page_index):03}.jpg')

            # Randomly assign dataset types ensuring they are all different but making it such that 70% of images in the train set are full images
            random_number = random.random()
            if random_number < 0.70:
                full_dataset_type = 'train'
                if random.random() < 0.75:
                    right_dataset_type = 'val'
                    left_dataset_type = 'test'
                else:
                    right_dataset_type = 'test'
                    left_dataset_type = 'val'
            else:
                full_dataset_type = 'val' if random.random() < 0.5 else 'test'
                right_dataset_type = 'train'
                left_dataset_type = 'train'

            # output paths
            full_image_output_path = f'Manga109_YOLO/{full_dataset_type}/images/{book_title}_{int(page_index):03}.jpg'
            right_image_output_path = f'Manga109_YOLO/{right_dataset_type}/images/{book_title}_{int(page_index):03}_right.jpg'
            left_image_output_path = f'Manga109_YOLO/{left_dataset_type}/images/{book_title}_{int(page_index):03}_left.jpg'

            # Copy/Create image to new directory
            shutil.copyfile(image_path, full_image_output_path)

            # Get the image dimensions
            width = int(page.get('width'))
            height = int(page.get('height'))

            # Create the right and left split images
            img = cv2.imread(image_path)
            cv2.imwrite(right_image_output_path, img[:, width // 2:])
            cv2.imwrite(left_image_output_path, img[:, :width // 2])

            # Open label files for full, right, and left images
            full_label_file_path = f'Manga109_YOLO/{full_dataset_type}/labels/{book_title}_{int(page_index):03}.txt'
            right_label_file_path = f'Manga109_YOLO/{right_dataset_type}/labels/{book_title}_{int(page_index):03}_right.txt'
            left_label_file_path = f'Manga109_YOLO/{left_dataset_type}/labels/{book_title}_{int(page_index):03}_left.txt'

            with open(full_label_file_path, 'w') as full_label_file, \
                    open(right_label_file_path, 'w') as right_label_file, \
                    open(left_label_file_path, 'w') as left_label_file:

                # Loop through annotations and write to full, right, and left labels
                for i, frame_type in enumerate(['face', 'body', 'text', 'frame']):
                    for frame in page.findall(frame_type):
                        xmin = int(frame.get('xmin'))
                        xmax = int(frame.get('xmax'))
                        ymin = int(frame.get('ymin'))
                        ymax = int(frame.get('ymax'))

                        # Normalized height and y_center (common for both halves)
                        norm_height = (ymax - ymin) / height
                        norm_y_center = (ymin + ymax) / (2 * height)

                        # Full image normalized coordinates
                        norm_x_center_full = (xmin + xmax) / (2 * width)
                        norm_width_full = (xmax - xmin) / width
                        full_label_file.write(f'{i} {norm_x_center_full} {norm_y_center} {norm_width_full} {norm_height}\n')

                        # Handle the right half
                        if xmin >= width // 2:
                            norm_x_center_right = ((xmin - width // 2) + (xmax - width // 2)) / (2 * (width // 2))
                            norm_width_right = (xmax - xmin) / (width // 2)
                            right_label_file.write(f'{i} {norm_x_center_right} {norm_y_center} {norm_width_right} {norm_height}\n')

                        # Handle the left half
                        elif xmax <= width // 2:
                            norm_x_center_left = (xmin + xmax) / (2 * (width // 2))
                            norm_width_left = (xmax - xmin) / (width // 2)
                            left_label_file.write(f'{i} {norm_x_center_left} {norm_y_center} {norm_width_left} {norm_height}\n')

                        # Handle annotations spanning both halves
                        else:
                            # Right portion
                            norm_x_center_right = ((0) + (xmax - width // 2)) / (2 * (width // 2))
                            norm_width_right = (xmax - width // 2) / (width // 2)
                            right_label_file.write(f'{i} {norm_x_center_right} {norm_y_center} {norm_width_right} {norm_height}\n')

                            # Left portion
                            norm_x_center_left = (xmin + (width // 2)) / (2 * (width // 2))
                            norm_width_left = (width // 2 - xmin) / (width // 2)
                            left_label_file.write(f'{i} {norm_x_center_left} {norm_y_center} {norm_width_left} {norm_height}\n')

            IMG_COUNTER += 1  # Increment the counter
            if IMG_COUNTER % 50 == 0:
                print(f'Processed {IMG_COUNTER} images')

    return 0



def main():
    create_dataset_yolo()

    return 0

if __name__ == '__main__':
    main()
