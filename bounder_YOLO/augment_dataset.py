import os
import yaml
import utils
import random
import albumentations as A


def read_augmentation_settings(yaml_path):
    # Read the YAML file with augmentation settings and return as a dictionary
    with open(yaml_path, 'r') as file:
        augmentation_settings = yaml.safe_load(file)

    return augmentation_settings


def random_transform(augmentation_settings, p=1.0):
    # Create a list of albumentations transformations based on the settings
    transforms = list(augmentation_settings.keys())
    random.shuffle(transforms)
    augmentation_list = []

    for transform in transforms:
        if transform == 'hsv_h' and 'hsv_s' in transforms and 'hsv_v' in transforms:
            augmentation_list.append(A.HueSaturationValue(
                hue_shift_limit=int(augmentation_settings['hsv_h'] * 255),
                sat_shift_limit=int(augmentation_settings['hsv_s'] * 255),
                val_shift_limit=int(augmentation_settings['hsv_v'] * 255),
                p=p
            ))
        elif transform == 'degrees':
            augmentation_list.append(A.Rotate(limit=augmentation_settings['degrees'], p=p))
        elif transform == 'translate':
            augmentation_list.append(A.ShiftScaleRotate(shift_limit=augmentation_settings['translate'], p=p))
        elif transform == 'scale':
            augmentation_list.append(A.RandomScale(scale_limit=augmentation_settings['scale'], p=p))
        elif transform == 'perspective':
            augmentation_list.append(A.Perspective(scale=augmentation_settings['perspective'], p=p))
        elif transform == 'flipud' and augmentation_settings['flipud'] > 0:
            augmentation_list.append(A.VerticalFlip(p=augmentation_settings['flipud']))
        elif transform == 'fliplr' and augmentation_settings['fliplr'] > 0:
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
    transform = A.Compose(augmentations, bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
    transformed = transform(image=image, bboxes=bboxes)
    image, annotations_raw = transformed['image'], transformed['bboxes']

    # remove non axis aligned bounding boxes
    annotations_raw = [bbox for bbox in annotations_raw if bbox[2] > 0 and bbox[3] > 0]

    # Convert transformed bounding boxes back to a dictionary of class annotations
    annotations = {class_type: [] for class_type in utils.ID_CLASS_MAP.values()}
    for bbox in annotations_raw:
        x_center, y_center, width, height, class_type = bbox
        annotations[utils.ID_CLASS_MAP[class_type]].append([x_center, y_center, width, height])

    return image, annotations


def main():
    yaml_path = 'augmentations.yaml'
    # augmentation_settings = read_augmentation_settings(yaml_path)

    augmentation_settings = {
        # 'hsv_h': 1,
        # 'hsv_s': 1,
        # 'hsv_v': 1,
        'degrees': 180.0,
        # 'translate': .2,
        # 'scale': 0.5,
        # 'shear': 2.0,
        # 'perspective': 1.0,
        # 'flipud': 0.5,
        # 'fliplr': 0.5
    }
    transform = random_transform(augmentation_settings)

    image_path = "Manga109_YOLO/train/images/AisazuNihaIrarenai_020_left.jpg"
    image, labels = utils.get_data_yolo(image_path)
    # utils.show_annotated_img_yolo(image, labels)
    
    transformed_image, transformed_bboxes = augment(image, labels, transform)
    utils.show_annotated_img_yolo(transformed_image, transformed_bboxes)


if __name__ == '__main__':
    main()