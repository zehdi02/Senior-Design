import os
import cv2
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Grayscale, ToTensor, Resize, Normalize


class Manga109Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = Compose([
            Resize((1024, 1024)),
            Grayscale(),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5])
        ])
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        images_dir = os.path.join(self.root_dir, 'images')
        annotations_dir = os.path.join(self.root_dir, 'annotations')

        for annotation_file in os.listdir(annotations_dir):
            tree = ET.parse(os.path.join(annotations_dir, annotation_file))
            root = tree.getroot()
            book_title = root.get('title')

            for page in root.findall('.//page'):
                page_index = page.get('index')
                image_path = os.path.join(images_dir, book_title, f'{int(page_index):03}.jpg')

                bboxes = []
                width = int(page.get('width'))
                height = int(page.get('height'))

                for frame in page.findall('frame'):
                    bbox = (
                        int(frame.get('xmin')) / width,
                        int(frame.get('ymin')) / height,
                        int(frame.get('xmax')) / width,
                        int(frame.get('ymax')) / height,
                    )
                    bboxes.append(bbox)

                if bboxes:
                    samples.append((image_path, bboxes))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, bboxes = self.samples[idx]
        image = Image.open(image_path)
        image_np = np.array(image)
        edges = cv2.Canny(image_np, 100, 200)
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        filled_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)
        image = Image.fromarray(filled_edges)
        image = self.transform(image)

        GRID_SIZE = 8
        y_true = torch.zeros((GRID_SIZE, GRID_SIZE, 4)) # 5 if conf

        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox

            x_center = ((xmin + xmax) / 2)
            y_center = ((ymin + ymax) / 2)
            width = (xmax - xmin)
            height = (ymax - ymin)

            # Calculate grid cell indices for the center
            center_col = int(x_center * GRID_SIZE)
            center_row = int(y_center * GRID_SIZE)

            y_true[center_col, center_row, :] = torch.tensor([x_center, y_center, width, height]) # , 1]) # if conf

        return image, y_true
