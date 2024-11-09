import os
from idlelib.pyparse import trans

import yaml
from numpy.lib.twodim_base import fliplr

import utils
import random
from tqdm import tqdm
import albumentations as A
from concurrent.futures import ThreadPoolExecutor, as_completed


def random_transform(augmentation_settings, p=1):
    """
    Randomly select and apply a set of transformations based on settings and probability.

    Args:
        augmentation_settings (dict): Settings specifying augmentation types and magnitudes.
        p (float): Probability of applying each transformation.

    Returns:
        list: A list of albumentations transformations.
    """
    # List of available transformation types from settings
    transforms = list(augmentation_settings.keys())
    random.shuffle(transforms)  # Randomize order of transformations
    augmentation_list = []

    # Iterate over each transform and decide to apply it based on probability `p`
    for transform in transforms:
        p_transform = 0 if random.random() > p else 1  # Apply transformation with a randomized probability

        # Map transform name to albumentations function, with parameters based on settings
        if transform == 'hsv_h':
            augmentation_list.append(
                A.HueSaturationValue(hue_shift_limit=int(augmentation_settings['hsv_h'] * 255), p=p_transform))
        elif transform == 'hsv_s':
            augmentation_list.append(
                A.HueSaturationValue(sat_shift_limit=int(augmentation_settings['hsv_s'] * 255), p=p_transform))
        elif transform == 'hsv_v':
            augmentation_list.append(
                A.HueSaturationValue(val_shift_limit=int(augmentation_settings['hsv_v'] * 255), p=p_transform))
        elif transform == 'degrees':
            augmentation_list.append(A.Rotate(limit=augmentation_settings['degrees'], p=p_transform))
        elif transform == 'translate':
            augmentation_list.append(A.ShiftScaleRotate(shift_limit=augmentation_settings['translate'], p=p_transform))
        elif transform == 'scale':
            augmentation_list.append(A.RandomScale(scale_limit=augmentation_settings['scale'], p=p_transform))
        elif transform == 'perspective':
            augmentation_list.append(A.Perspective(scale=augmentation_settings['perspective'], p=p_transform))
        elif transform == 'flipud':
            augmentation_list.append(A.VerticalFlip(p=augmentation_settings['flipud']))
        elif transform == 'fliplr':
            augmentation_list.append(A.HorizontalFlip(p=augmentation_settings['fliplr']))

    return augmentation_list


def augment(image, annotations, augmentations):
    """
    Apply transformations to an image and its annotations (bounding boxes).

    Args:
        image (np.ndarray): Image to transform.
        annotations (dict): Bounding box annotations by class.

    Returns:
        tuple: Transformed image and updated annotations dictionary.
    """
    # Collect bounding boxes and labels for transformation
    bboxes = []
    for class_type, class_annotations in annotations.items():
        for annotation in class_annotations:
            x_center, y_center, width, height = annotation
            class_id = utils.CLASS_ID_MAP[class_type]
            bboxes.append([x_center, y_center, width, height, class_id])

    # Apply augmentation transformations to image and bounding boxes
    transform = A.Compose(augmentations, bbox_params=A.BboxParams(format='yolo', min_visibility=0.6))
    transformed = transform(image=image, bboxes=bboxes)
    image, annotations_raw = transformed['image'], transformed['bboxes']

    # Convert transformed bounding boxes back into dictionary format
    annotations = {class_type: [] for class_type in utils.ID_CLASS_MAP.values()}
    for bbox in annotations_raw:
        x_center, y_center, width, height, class_type = bbox
        annotations[utils.ID_CLASS_MAP[class_type]].append([x_center, y_center, width, height])

    return image, annotations


def augment_yolo(image_path, augmentation_settings, p):
    """
    Load image and annotations, apply augmentations, and save the output.

    Args:
        image_path (str): Path to the input image.
        augmentation_settings (dict): Dictionary of augmentation settings.
        p (float): Probability of applying each augmentation.
    """
    image, labels = utils.get_data_yolo(image_path)
    transform_list = random_transform(augmentation_settings, p)
    transformed_image, transformed_bboxes = augment(image, labels, transform_list)

    # Define output paths for transformed image and label file
    image_name = os.path.basename(image_path)
    dataset_type = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    image_output_path = f'{utils.OUTPUT_DIR}/{dataset_type}_aug/images/{image_name}'
    label_file_path = f'{utils.OUTPUT_DIR}/{dataset_type}_aug/labels/{image_name[:-4]}.txt'

    # Save transformed image and bounding box annotations
    utils.save_image_and_labels(transformed_image, transformed_bboxes, image_output_path, label_file_path)

    return transformed_image, transformed_bboxes


def test_augmentation():
    """
    Load augmentation settings, create necessary directories, and test transformations.
    """
    yaml_path = 'augmentations.yaml'
    with open(yaml_path, 'r') as file:
        augmentation_settings = yaml.safe_load(file)

    augmentation_settings = {
        'flipud': 1,
        'fliplr': 1
    }

    # Create directories for augmented images and labels
    for dataset_type in ['train', 'val']:
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}_aug/images', exist_ok=True)
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}_aug/labels', exist_ok=True)

    # Test augmentation on sample images
    augment_yolo("Manga109_YOLO/train/images/AisazuNihaIrarenai_020_left.jpg", augmentation_settings, p=1)
    augment_yolo("Manga109_YOLO/train/images/AisazuNihaIrarenai_020_right.jpg", augmentation_settings, p=1)


def augment_dataset(augmentation_settings, p=1):
    """
    Iterate over train and val images, applying augmentations concurrently and saving results.

    Args:
        augmentation_settings (dict): Dictionary of augmentation settings.
        p (float): Probability of applying each transformation.
    """
    # Create directories for storing augmented dataset
    for dataset_type in ['train', 'val']:
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}_aug/images', exist_ok=True)
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}_aug/labels', exist_ok=True)

    # Process each dataset type concurrently
    for dataset_type in ['train', 'val']:
        image_dir = f'{utils.OUTPUT_DIR}/{dataset_type}/images'
        image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]

        # Using ThreadPoolExecutor to apply augmentations concurrently
        with ThreadPoolExecutor() as image_executor, tqdm(total=len(image_paths), desc="Images Processed") as pbar:
            futures = [image_executor.submit(augment_yolo, path, augmentation_settings, p) for path in image_paths]

            # Update progress bar upon completion of each image transformation
            for future in as_completed(futures):
                pbar.update(1)

    return


def main():
    # Define augmentation settings
    augmentation_settings = {
        # 'fliplr': 0.52756,
        # 'flipud': 0.0,
        # 'hsv_h': 0.01852,
        # 'hsv_s': 0.62426,
        # 'hsv_v': 0.30038,
        # 'translate': 0.09958,
        # 'scale': 0.34871,
        'flipud': 1,
        'fliplr': 1
    }
    # test_augmentation()  # Run test augmentation on sample images
    augment_dataset(augmentation_settings, p=1)


if __name__ == '__main__':
    main()
