import os
import shutil
import random
import xml.etree.ElementTree as ET


def create_dataset_yolo():
    counter = 0
    root_dir = '../Manga109'
    images_dir = os.path.join(root_dir, 'images')
    annotations_dir = os.path.join(root_dir, 'annotations')

    # Create train, val, test directories
    for dataset_type in ['train', 'val', 'test']:
        os.makedirs(f'../Manga109_YOLO/{dataset_type}/images', exist_ok=True)
        os.makedirs(f'../Manga109_YOLO/{dataset_type}/labels', exist_ok=True)

    for annotation_file in os.listdir(annotations_dir):
        tree = ET.parse(os.path.join(annotations_dir, annotation_file))
        root = tree.getroot()
        book_title = root.get('title')

        for page in root.findall('.//page'):
            page_index = page.get('index')
            image_path = os.path.join(images_dir, book_title, f'{int(page_index):03}.jpg')

            # Randomly assign to train/val/test and copy image + txt file to new dir
            dataset_type = random.choices(['train', 'val', 'test'], weights=[0.85, 0.1, 0.05])[0]

            # Copy the image to the new directory
            shutil.copyfile(image_path, f'../Manga109_YOLO/{dataset_type}/images/{book_title}_{int(page_index):03}.jpg')

            # Create a new text file in the labels directory for that chosen dataset
            label_file_path = f'../Manga109_YOLO/{dataset_type}/labels/{book_title}_{int(page_index):03}.txt'
            label_file = open(label_file_path, 'w')

            # populate text file with annotations
            width = int(page.get('width'))
            height = int(page.get('height'))

            for i, frame_type in enumerate(['face', 'body', 'text', 'frame']):
                for frame in page.findall(frame_type):
                    bbox = (
                        (int(frame.get('xmin')) + int(frame.get('xmax'))) / (2 * width),
                        (int(frame.get('ymin')) + int(frame.get('ymax'))) / (2 * height),
                        (int(frame.get('xmax')) - int(frame.get('xmin'))) / width,
                        (int(frame.get('ymax')) - int(frame.get('ymin'))) / height,
                    )
                    label_file.write(f'{i} {" ".join(map(str, bbox))}\n')
            label_file.close()

            counter += 1  # Increment the counter

            # Print the counter every 50 increments
            if counter % 50 == 0:
                print(f'Processed {counter} images')

    return 0


def main():
    create_dataset_yolo()

    return 0


if __name__ == '__main__':
    main()
