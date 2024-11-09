import os
import yaml
import random
from tqdm import tqdm
import albumentations as A
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils


def random_transform(augmentation_settings, p=1):
    """
    Generate a list of random transformations based on given settings.

    Args:
        augmentation_settings (dict): Dictionary specifying each augmentation type (like 'hsv_h', 'flipud')
                                      and its configuration values.
        p (float): Probability of applying each transformation. Value should be between 0 (never apply) and 1 (always apply).

    Returns:
        list: A list of albumentations transformations to be applied to an image.
    """
    # Collect available transformations from settings and shuffle their order for randomness
    transforms = list(augmentation_settings.keys())
    random.shuffle(transforms)
    augmentation_list = []

    # Add transformations to the list based on settings and probability
    for transform in transforms:
        # Determine whether to apply each transformation based on probability
        p_transform = 0 if random.random() > p else 1

        # Add transformation with specific settings based on type
        if transform == 'hsv_h':
            augmentation_list.append(A.HueSaturationValue(hue_shift_limit=int(augmentation_settings['hsv_h'] * 255), p=p_transform))
        elif transform == 'hsv_s':
            augmentation_list.append(A.HueSaturationValue(sat_shift_limit=int(augmentation_settings['hsv_s'] * 255), p=p_transform))
        elif transform == 'hsv_v':
            augmentation_list.append(A.HueSaturationValue(val_shift_limit=int(augmentation_settings['hsv_v'] * 255), p=p_transform))
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
    Apply augmentations to an image and its bounding box annotations.

    Args:
        image (np.ndarray): The image to augment.
        annotations (dict): Dictionary of bounding box annotations, organized by class.
        augmentations (list): List of albumentations transformations to apply.

    Returns:
        tuple: A tuple containing:
            - transformed image (np.ndarray)
            - transformed annotations (dict) with updated bounding boxes.
    """
    # Convert annotations to list of bounding boxes with labels
    bboxes = []
    for class_type, class_annotations in annotations.items():
        for annotation in class_annotations:
            x_center, y_center, width, height = annotation
            class_id = utils.CLASS_ID_MAP[class_type]  # Map class name to numeric ID
            bboxes.append([x_center, y_center, width, height, class_id])

    # Compose transformation pipeline and apply to both image and bounding boxes
    transform = A.Compose(augmentations, bbox_params=A.BboxParams(format='yolo', min_visibility=0.6))
    transformed = transform(image=image, bboxes=bboxes)
    image, annotations_raw = transformed['image'], transformed['bboxes']

    # Convert transformed bounding boxes back into dictionary format by class
    annotations = {class_type: [] for class_type in utils.ID_CLASS_MAP.values()}
    for bbox in annotations_raw:
        x_center, y_center, width, height, class_type = bbox
        annotations[utils.ID_CLASS_MAP[class_type]].append([x_center, y_center, width, height])

    return image, annotations


def augment_yolo(image_path, augmentation_settings, p):
    """
    Load an image and annotations, apply augmentations, and save the augmented output.

    Args:
        image_path (str): Path to the input image file.
        augmentation_settings (dict): Dictionary of augmentation settings.
        p (float): Probability of applying each augmentation.

    Returns:
        tuple: Augmented image and updated bounding box annotations.
    """
    # Load the image and annotations from the dataset
    image, labels = utils.get_data_yolo(image_path)

    # Generate list of transformations and apply to image and bounding boxes
    transform_list = random_transform(augmentation_settings, p)
    transformed_image, transformed_bboxes = augment(image, labels, transform_list)

    # Prepare output paths based on input path
    image_name = os.path.basename(image_path)
    dataset_type = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    image_output_path = f'{utils.OUTPUT_DIR}/{dataset_type}_aug/images/{image_name}'
    label_file_path = f'{utils.OUTPUT_DIR}/{dataset_type}_aug/labels/{image_name[:-4]}.txt'

    # Save the augmented image and corresponding label file
    utils.save_image_and_labels(transformed_image, transformed_bboxes, image_output_path, label_file_path)

    return transformed_image, transformed_bboxes


def test_augmentation():
    """
    Test augmentation function on sample images, creating output directories as needed.
    """
    # Load or define augmentation settings
    yaml_path = 'augmentations.yaml'
    with open(yaml_path, 'r') as file:
        augmentation_settings = yaml.safe_load(file)

    augmentation_settings = {
        'flipud': 1,
        'fliplr': 1
    }

    # Ensure output directories for augmented images and labels exist
    for dataset_type in ['train', 'val']:
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}_aug/images', exist_ok=True)
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}_aug/labels', exist_ok=True)

    # Run sample augmentations on test images
    augment_yolo("Manga109_YOLO/train/images/AisazuNihaIrarenai_020_left.jpg", augmentation_settings, p=1)
    augment_yolo("Manga109_YOLO/train/images/AisazuNihaIrarenai_020_right.jpg", augmentation_settings, p=1)


def augment_dataset(augmentation_settings, p=1):
    """
    Apply augmentations to all images in the 'train' and 'val' directories, saving results concurrently.

    Args:
        augmentation_settings (dict): Dictionary of settings for various augmentations.
        p (float): Probability of applying each transformation.

    Returns:
        None
    """
    # Ensure directories for augmented images and labels exist
    for dataset_type in ['train', 'val']:
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}_aug/images', exist_ok=True)
        os.makedirs(f'{utils.OUTPUT_DIR}/{dataset_type}_aug/labels', exist_ok=True)

    # Process images in each dataset type directory
    for dataset_type in ['train', 'val']:
        image_dir = f'{utils.OUTPUT_DIR}/{dataset_type}/images'
        image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]

        # Use ThreadPoolExecutor to apply augmentations concurrently, improving speed
        with ThreadPoolExecutor() as image_executor, tqdm(total=len(image_paths), desc=f"Processing {dataset_type} images") as pbar:
            futures = [image_executor.submit(augment_yolo, path, augmentation_settings, p) for path in image_paths]

            # Progress bar updates after each image completes augmentation
            for future in as_completed(futures):
                future.result()  # Retrieve result to handle exceptions in concurrent execution
                pbar.update(1)  # Update progress bar by one unit per completed future


def main():
    """
    Main function to either test sample augmentations or apply augmentations to the full dataset.
    """
    test_augmentation()  # Run sample test augmentations

    augmentation_settings = {
        'flipud': 1,
        'fliplr': 1
    }
    # augment_dataset(augmentation_settings, p=1)


if __name__ == '__main__':
    main()
