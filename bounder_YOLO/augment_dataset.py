import os
from idlelib.pyparse import trans

import yaml
import utils
import random
from tqdm import tqdm
import albumentations as A
from concurrent.futures import ThreadPoolExecutor, as_completed


def random_transform(augmentation_settings, p=1):
    """
    Randomly select a set of transformations based on the settings and probability.

    Args:
        augmentation_settings (dict): Dictionary of augmentation settings.
        p (float): Probability of applying the transformation.

    Returns:
        list: A list of albumentations transformations.
    """
    # Create a list of albumentations transformations based on the settings
    transforms = list(augmentation_settings.keys())
    random.shuffle(transforms)
    augmentation_list = []

    for transform in transforms:
        p = 0 if random.random() > p else 1

        if transform == 'hsv_h':
            augmentation_list.append(
                A.HueSaturationValue(hue_shift_limit=int(augmentation_settings['hsv_h'] * 255), p=p))

        elif transform == 'hsv_s':
            augmentation_list.append(
                A.HueSaturationValue(sat_shift_limit=int(augmentation_settings['hsv_s'] * 255), p=p))

        elif transform == 'hsv_v':
            augmentation_list.append(
                A.HueSaturationValue(val_shift_limit=int(augmentation_settings['hsv_v'] * 255), p=p))

        elif transform == 'degrees':
            augmentation_list.append(A.Rotate(limit=augmentation_settings['degrees'], p=p))

        elif transform == 'translate':
            augmentation_list.append(A.ShiftScaleRotate(shift_limit=augmentation_settings['translate'], p=p))

        elif transform == 'scale':
            augmentation_list.append(A.RandomScale(scale_limit=augmentation_settings['scale'], p=p))

        elif transform == 'perspective':
            augmentation_list.append(A.Perspective(scale=augmentation_settings['perspective'], p=p))

        elif transform == 'flipud':
            augmentation_list.append(A.VerticalFlip(p=augmentation_settings['flipud']))

        elif transform == 'fliplr':
            augmentation_list.append(A.HorizontalFlip(p=augmentation_settings['fliplr']))

    return augmentation_list


def augment(image, annotations, augmentations):
    """
    Split an image and its annotations into left and right halves.

    Args:
        image (np.ndarray): Image to split.
        annotations (dict): Dictionary of annotations by class in xyxy format.

    Returns:
        tuple: A tuple containing the left image, left annotations, right image, and right annotations.
    """
    # create list of annotations and their respective class labels
    bboxes = []
    for class_type, class_annotations in annotations.items():
        for annotation in class_annotations:
            x_center, y_center, width, height = annotation
            class_id = utils.CLASS_ID_MAP[class_type]
            bboxes.append([x_center, y_center, width, height, class_id])

    # Transform image and bounding boxes
    transform = A.Compose(augmentations, bbox_params=A.BboxParams(format='yolo', min_visibility=0.6))
    transformed = transform(image=image, bboxes=bboxes)
    image, annotations_raw = transformed['image'], transformed['bboxes']

    # Convert transformed bounding boxes back to a dictionary of class annotations
    annotations = {class_type: [] for class_type in utils.ID_CLASS_MAP.values()}
    for bbox in annotations_raw:
        x_center, y_center, width, height, class_type = bbox
        annotations[utils.ID_CLASS_MAP[class_type]].append([x_center, y_center, width, height])

    return image, annotations


def augment_yolo(image_path, augmentation_settings, p):
    image, labels = utils.get_data_yolo(image_path)
    transform_list = random_transform(augmentation_settings, p)
    transformed_image, transformed_bboxes = augment(image, labels, transform_list)

    image = image_path.split('/')[-1]
    dataset_type = image_path.split('/')[-3]
    image_output_path = f'{utils.OUTPUT_DIR}/{dataset_type}_aug/images/{image}'
    label_file_path = f'{utils.OUTPUT_DIR}/{dataset_type}_aug/labels/{image[:-4]}.txt'
    utils.save_image_and_labels(transformed_image, transformed_bboxes, image_output_path, label_file_path)

    return transformed_image, transformed_bboxes


def test_augmentation():
    yaml_path = 'augmentations.yaml'
    with open(yaml_path, 'r') as file:
        augmentation_settings = yaml.safe_load(file)

    augmentation_settings = {
        # 'hsv_v': 1,
        # 'hsv_h': 1,
        # 'hsv_s': 1,
        # 'degrees': 180.0,
        # 'translate': .2,
        # 'scale': 1,
        # 'shear': 2.0,
        # 'perspective': .1,
        'flipud': 1,
        'fliplr': 1
    }
    # transform = random_transform(augmentation_settings)

    for dataset_type in ['train', 'val']:
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}_aug/images', exist_ok=True)
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}_aug/labels', exist_ok=True)

    image_path = "Manga109_YOLO/train/images/AisazuNihaIrarenai_020_left.jpg"

    # image, labels = utils.get_data_yolo(image_path)
    # # utils.show_annotated_img_yolo(image, labels)
    # transformed_image, transformed_bboxes = augment(image, labels, transform)
    # utils.show_annotated_img_yolo(transformed_image, transformed_bboxes)

    augment_yolo(image_path, augmentation_settings, p=1)


    image_path = "Manga109_YOLO/train/images/AisazuNihaIrarenai_020_right.jpg"

    # image, labels = utils.get_data_yolo(image_path)
    # # utils.show_annotated_img_yolo(image, labels)
    # transformed_image, transformed_bboxes = augment(image, labels, transform)
    # utils.show_annotated_img_yolo(transformed_image, transformed_bboxes)

    augment_yolo(image_path, augmentation_settings, p=1)



def augment_dataset(augmentation_settings, p=1):
    """
    Go through each image in the train and val dirs and augment them with given settings and probability by overwriting them with augmented result of their original in Mangaa109 dataset

    Args:
        augmentation_settings (dict): Dictionary of augmentation settings.
        p (float): Probability of applying the transformation.
    """
    # create new directories for augmented images and labels
    for dataset_type in ['train', 'val']:
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}_aug/images', exist_ok=True)
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}_aug/labels', exist_ok=True)

    # go through each image in the train and val dirs and save augmented images and labels in the new directories
    for dataset_type in ['train', 'val']:
        with tqdm(total=len(os.listdir(f'{utils.OUTPUT_DIR}/{dataset_type}/images')), desc="Images Processes") as pbar:
            with ThreadPoolExecutor() as image_executor:
                image_futures =[]
                for image_path in os.listdir(f'{utils.OUTPUT_DIR}/{dataset_type}/images'):
                    image_futures.append(image_executor.submit(augment_yolo, f'{utils.OUTPUT_DIR}/{dataset_type}/images/{image_path}', augmentation_settings, p))
                    pbar.update(1)
    return


def main():
    augmentation_settings = {
        'flipud': 1,
        'fliplr': 1
    }
    # augment_dataset(augmentation_settings, p=1)
    test_augmentation()


if __name__ == '__main__':
    main()
