import os
import cv2
import shutil
import random
import xml.etree.ElementTree as ET
import numpy as np  # Import numpy for arrays

ROOT_DIR = '../Manga109'
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, 'annotations')
DATASET_TYPES = ['train', 'val', 'test']

def augment_image_and_labels(image, labels):
    def rotate_labels(labels, angle, w, h):
        print(labels[0])
        center_x, center_y = w / 2, h / 2
        angle_rad = np.radians(angle)

        for label in labels:
            # get the corners of the bounding box
            x_center = label[1] * w
            y_center = label[2] * h
            width = label[3] * w
            height = label[4] * h

            # get the corners of the bounding box
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # rotate the corners
            x1r = center_x + np.cos(angle_rad) * (x1 - center_x) - np.sin(angle_rad) * (y1 - center_y)
            y1r = center_y + np.sin(angle_rad) * (x1 - center_x) + np.cos(angle_rad) * (y1 - center_y)
            x2r = center_x + np.cos(angle_rad) * (x2 - center_x) - np.sin(angle_rad) * (y2 - center_y)
            y2r = center_y + np.sin(angle_rad) * (x2 - center_x) + np.cos(angle_rad) * (y2 - center_y)

            # get the enclosing box (min x, min y, max x, max y) which is the new bounding box
            new_x_center = (x1r + x2r) / 2
            new_y_center = (y1r + y2r) / 2
            new_width = x2r - x1r
            new_height = y2r - y1r

            # update the label with the new coordinates and dimensions
            label[1] = new_x_center / w
            label[2] = new_y_center / h
            label[3] = new_width / w
            label[4] = new_height / h

        return labels

    def translate_labels(labels, tx, ty):
        print(labels[0])
        for label in labels:
            label[1] += tx / w
            label[2] += ty / h
        print(labels[0])
        return labels

    def perspective_labels(labels, M, w, h):
        print(labels[0])
        for i in range(len(labels)):
            label = labels[i]

            x_center = label[1] * w
            y_center = label[2] * h
            width = label[3] * w
            height = label[4] * h

            corners = np.array([
                [x_center - width / 2, y_center - height / 2],
                [x_center + width / 2, y_center - height / 2],
                [x_center + width / 2, y_center + height / 2],
                [x_center - width / 2, y_center + height / 2]
            ])

            new_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), M).reshape(-1, 2)

            new_x_center = np.mean(new_corners[:, 0])
            new_y_center = np.mean(new_corners[:, 1])
            new_width = np.max(new_corners[:, 0]) - np.min(new_corners[:, 0])
            new_height = np.max(new_corners[:, 1]) - np.min(new_corners[:, 1])

            # Update label with new coordinates and dimensions
            label[1] = new_x_center / w
            label[2] = new_y_center / h
            label[3] = new_width / w
            label[4] = new_height / h

            labels[i] = label
        print(labels[0])

        return labels

    def scale_labels(labels, scale_factor):
        print(labels[0])

        for label in labels:
            label[1] *= scale_factor
            label[2] *= scale_factor
            label[3] *= scale_factor
            label[4] *= scale_factor
        print(labels[0])

        return labels

    def shear_labels(labels, shear_factor):
        print(labels[0])

        for label in labels:
            x_center = label[1] * w
            y_center = label[2] * h

            new_x_center = x_center + shear_factor * (y_center - (h / 2))
            label[1] = new_x_center / w

        print(labels[0])

        return labels

    def flip_labels(labels):
        print(labels[0])

        for label in labels:
            label[1] = 1 - label[1]  # Flip the x-coordinate
        print(labels[0])
        return labels

    if len(image.shape) == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    augmentations = [
        ('rotate', random.uniform(0, 90))
        # ('translate', (random.randint(-w // 4, w // 4), random.randint(-h // 4, h // 4))),
        # ('perspective', True),
        # ('scale', random.uniform(0.5, 1.5)),
        # ('shear', random.uniform(-0.2, 0.2)),
        # ('flip', random.choice([True, False])),
    ]

    for aug in augmentations:
        if random.random() < 1:  # 10% chance to apply each augmentation
            if aug[0] == 'rotate':
                print("rotate")
                angle = aug[1]
                h, w = gray_image.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                gray_image = cv2.warpAffine(gray_image, M, (w, h))
                labels = rotate_labels(labels, angle, w, h)

            elif aug[0] == 'translate':
                print("translate")
                tx, ty = aug[1]
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                gray_image = cv2.warpAffine(gray_image, M, (gray_image.shape[1], gray_image.shape[0]))
                labels = translate_labels(labels, tx, ty)

            elif aug[0] == 'perspective':
                print("perspective")
                rows, cols = gray_image.shape[:2]
                pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
                pts2 = np.float32([[random.randint(-10, 10), random.randint(-10, 10)],
                                   [cols + random.randint(-10, 10), random.randint(-10, 10)],
                                   [random.randint(-10, 10), rows + random.randint(-10, 10)],
                                   [cols + random.randint(-10, 10), rows + random.randint(-10, 10)]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                gray_image = cv2.warpPerspective(gray_image, M, (cols, rows))
                labels = perspective_labels(labels, M, w, h)

            elif aug[0] == 'scale':
                print("scale")
                scale_factor = aug[1]
                gray_image = cv2.resize(gray_image, None, fx=scale_factor, fy=scale_factor)
                labels = scale_labels(labels, scale_factor)

            elif aug[0] == 'shear':
                print("shear")
                shear_factor = aug[1]
                M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
                gray_image = cv2.warpAffine(gray_image, M, (gray_image.shape[1], gray_image.shape[0]))
                labels = shear_labels(labels, shear_factor)

            elif aug[0] == 'flip' and aug[1]:
                print("flip")
                gray_image = cv2.flip(gray_image, 1)
                labels = flip_labels(labels)

    return gray_image, labels


def create_dataset_yolo():
    IMG_COUNTER = 0

    # Create train, val, test directories
    for dataset_type in DATASET_TYPES:
        os.makedirs(f'Manga109_YOLO/{dataset_type}/images', exist_ok=True)
        os.makedirs(f'Manga109_YOLO/{dataset_type}/labels', exist_ok=True)

    # Loop through all the annotation files
    for annotation_file in os.listdir(ANNOTATIONS_DIR):
        if IMG_COUNTER == 5:
            break
        tree = ET.parse(os.path.join(ANNOTATIONS_DIR, annotation_file))
        root = tree.getroot()
        book_title = root.get('title')
        print(f'Found book title: {book_title}')

        # Loop through pages in the annotation file
        for page in root.findall('.//page'):
            if IMG_COUNTER == 5:
                break
            page_index = page.get('index')
            image_path = os.path.join(IMAGES_DIR, book_title, f'{int(page_index):03}.jpg')

            # Randomly assign dataset types ensuring they are all different
            random_number = random.random()
            if random_number < 0.80:
                full_dataset_type = 'train'
                right_dataset_type = 'val' if random.random() < 0.50 else 'test'
                left_dataset_type = 'test' if right_dataset_type == 'val' else 'val'
            else:
                full_dataset_type = 'val' if random.random() < 0.5 else 'test'
                right_dataset_type = 'train'
                left_dataset_type = 'train'

            # Output paths
            full_image_output_path = f'Manga109_YOLO/{full_dataset_type}/images/{book_title}_{int(page_index):03}.jpg'
            right_image_output_path = f'Manga109_YOLO/{right_dataset_type}/images/{book_title}_{int(page_index):03}_right.jpg'
            left_image_output_path = f'Manga109_YOLO/{left_dataset_type}/images/{book_title}_{int(page_index):03}_left.jpg'

            # Get the image dimensions
            width = int(page.get('width'))
            height = int(page.get('height'))

            # Create lists to store label data for full, right, and left images
            full_labels = []
            right_labels = []
            left_labels = []

            # Loop through annotations and store in the lists
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
                    full_labels.append([i, norm_x_center_full, norm_y_center, norm_width_full, norm_height])

                    # Handle the right half
                    if xmin >= width // 2:
                        norm_x_center_right = ((xmin - width // 2) + (xmax - width // 2)) / (2 * (width // 2))
                        norm_width_right = (xmax - xmin) / (width // 2)
                        right_labels.append([i, norm_x_center_right, norm_y_center, norm_width_right, norm_height])
                    # Handle the left half
                    elif xmax <= width // 2:
                        norm_x_center_left = (xmin + xmax) / (2 * (width // 2))
                        norm_width_left = (xmax - xmin) / (width // 2)
                        left_labels.append([i, norm_x_center_left, norm_y_center, norm_width_left, norm_height])
                    # Handle annotations spanning both halves
                    else:
                        # Right portion
                        norm_x_center_right = ((0) + (xmax - width // 2)) / (2 * (width // 2))
                        norm_width_right = (xmax - width // 2) / (width // 2)
                        right_labels.append([i, norm_x_center_right, norm_y_center, norm_width_right, norm_height])
                        # Left portion
                        norm_x_center_left = (xmin + (width // 2)) / (2 * (width // 2))
                        norm_width_left = (width // 2 - xmin) / (width // 2)
                        left_labels.append([i, norm_x_center_left, norm_y_center, norm_width_left, norm_height])

            # Read the original image and create and augment the right and left split images
            img = cv2.imread(image_path)
            right_img = img[:, width // 2:]
            left_img = img[:, :width // 2]

            # find which image is going to the training set
            if full_dataset_type == 'train' and len(full_labels) > 0:
                img, full_labels = augment_image_and_labels(img, full_labels)
            if right_dataset_type == 'train' and len(right_labels) > 0:
                right_img, right_labels = augment_image_and_labels(right_img, right_labels)
            if left_dataset_type == 'train' and len(left_labels) > 0:
                left_img, left_labels = augment_image_and_labels(left_img, left_labels)

            # Write images and labels to files after collecting all annotations
            cv2.imwrite(full_image_output_path, img)
            cv2.imwrite(right_image_output_path, right_img)
            cv2.imwrite(left_image_output_path, left_img)

            full_label_file_path = f'Manga109_YOLO/{full_dataset_type}/labels/{book_title}_{int(page_index):03}.txt'
            right_label_file_path = f'Manga109_YOLO/{right_dataset_type}/labels/{book_title}_{int(page_index):03}_right.txt'
            left_label_file_path = f'Manga109_YOLO/{left_dataset_type}/labels/{book_title}_{int(page_index):03}_left.txt'

            with open(full_label_file_path, 'w') as full_label_file, \
                    open(right_label_file_path, 'w') as right_label_file, \
                    open(left_label_file_path, 'w') as left_label_file:

                # Write to full labels
                for label in full_labels:
                    full_label_file.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

                # Write to right labels
                for label in right_labels:
                    right_label_file.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

                # Write to left labels
                for label in left_labels:
                    left_label_file.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

            IMG_COUNTER += 1  # Increment the counter
            if IMG_COUNTER % 50 == 0:
                print(f'Processed {IMG_COUNTER} images')

    return 0


def main():
    create_dataset_yolo()

    return 0

if __name__ == '__main__':
    main()
