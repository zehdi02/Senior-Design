import logging
import os
import random
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.ops import box_iou
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

                samples.append((image_path, bboxes))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, bboxes = self.samples[idx]
        image = Image.open(image_path).convert('L')
        image = self.transform(image)

        GRID_SIZE = 8
        y_true = torch.zeros((GRID_SIZE, GRID_SIZE, 5))

        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox

            x_center = ((xmin + xmax) / 2)
            y_center = ((ymin + ymax) / 2)
            width = (xmax - xmin)
            height = (ymax - ymin)

            # Calculate grid cell indices for the center
            center_col = int(x_center * GRID_SIZE)
            center_row = int(y_center * GRID_SIZE)

            # Determine the range of grid cells covered by the bbox
            start_col = int(xmin * GRID_SIZE)
            end_col = int(xmax * GRID_SIZE)
            start_row = int(ymin * GRID_SIZE)
            end_row = int(ymax * GRID_SIZE)

            # Iterate over the cells covered by the bbox
            for row in range(max(0, start_row), min(GRID_SIZE, end_row + 1)):
                for col in range(max(0, start_col), min(GRID_SIZE, end_col + 1)):
                    cell_xmin = col / GRID_SIZE
                    cell_xmax = (col + 1) / GRID_SIZE
                    cell_ymin = row / GRID_SIZE
                    cell_ymax = (row + 1) / GRID_SIZE

                    # Calculate intersection area
                    inter_xmin = max(xmin, cell_xmin)
                    inter_xmax = min(xmax, cell_xmax)
                    inter_ymin = max(ymin, cell_ymin)
                    inter_ymax = min(ymax, cell_ymax)
                    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

                    # Calculate cell area and bbox area
                    cell_area = (cell_xmax - cell_xmin) * (cell_ymax - cell_ymin)
                    bbox_area = (xmax - xmin) * (ymax - ymin)

                    # Set confidence based on overlap
                    if row == center_row and col == center_col:
                        confidence = 1.0  # Center cell
                    elif inter_area == cell_area:
                        confidence = 0.8  # Cell fully within the bbox
                    else:
                        confidence = 0.5  # Cell partially covered

                    y_true[row, col, :] = torch.tensor([x_center, y_center, width, height, confidence])

        return image, y_true


class CustomYOLO(nn.Module):
    def __init__(self):
        super(CustomYOLO, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=1),

            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(2, stride=1),

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, stride=1),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Upsample((448, 448), mode='bilinear', align_corners=False),

            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
        )

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(64, 192, 3)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.convs = nn.Sequential(
            nn.Conv2d(192, 128, 1),
            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 512, 3, padding=1),
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.convs2 = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(512, 256, 1),
                            nn.Conv2d(256, 512, 3, padding=1)) for i in range(4)],
            nn.Conv2d(512, 512, 1),
            nn.Conv2d(512, 1024, 3, padding=1),
        )
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.convs3 = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(1024, 512, 1, padding=1),
                            nn.Conv2d(512, 1024, 3, padding=1)) for i in range(2)],
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.Conv2d(1024, 1024, 3),
            nn.Conv2d(1024, 1024, 3),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(7 * 7 * 1024, 4096)
        self.fc2 = nn.Linear(4096, 8 * 8 * 5)

    def forward(self, x):
        x = self.reduce(x)
        assert not torch.isnan(x).any(), "reduce"

        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.pool3(F.leaky_relu(self.convs(x)))
        x = self.pool4(F.leaky_relu(self.convs2(x)))
        x = self.convs3(x)

        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = x.view(-1, 8, 8, 5)
        x = torch.sigmoid(x)

        return x


class YOLOLoss(nn.Module):
    def __init__(self, l_coord=2, l_iou=5, l_noobj=.01):
        super(YOLOLoss, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.l_iou = l_iou

    def forward(self, predictions, target):
        # Masks where target boxes are present or absent
        obj_mask = target[..., 4] > 0
        noobj_mask = target[..., 4] == 0

        # Compute losses for the bounding box coordinates (only for object-containing cells)
        box_predictions = predictions[..., :4][obj_mask]
        box_targets = target[..., :4][obj_mask]

        # Calculate IoU loss
        iou_scores = self.calculate_iou(box_predictions, box_targets)
        iou_loss = 1 - iou_scores.mean()  # Average IoU loss across all positive samples

        # MSE loss for bounding box coordinates
        box_loss = F.mse_loss(torch.sqrt(box_predictions[..., 2:4]), torch.sqrt(box_targets[..., 2:4]), reduction='sum')
        center_loss = F.mse_loss(box_predictions[..., :2], box_targets[..., :2], reduction='sum')

        # Compute the loss for confidence scores
        obj_confidence_loss = F.mse_loss(predictions[..., 4][obj_mask], target[..., 4][obj_mask], reduction='sum')
        noobj_confidence_loss = F.mse_loss(predictions[..., 4][noobj_mask], target[..., 4][noobj_mask], reduction='sum')

        # Calculate the total loss incorporating the differential weighting
        total_loss = (
                (self.l_coord * (box_loss + center_loss)) +
                (self.l_iou * iou_loss) + obj_confidence_loss +
                (self.l_noobj * noobj_confidence_loss)) / predictions.size(0)

        print(f"iou: {iou_loss}", f"box: {box_loss}", f"center: {center_loss}", f"obj_conf {obj_confidence_loss}", f"no_obj_conf {noobj_confidence_loss}")
        return total_loss

    def convert_cxcywh_to_xyxy(self, boxes):
        cx, cy, w, h = boxes.unbind(-1)
        xmin = cx - (w * 0.5)
        xmax = cx + (w * 0.5)
        ymin = cy - (h * 0.5)
        ymax = cy + (h * 0.5)
        return torch.stack((xmin, ymin, xmax, ymax), dim=-1)

    def calculate_iou(self, boxes1, boxes2):
        box1 = self.convert_cxcywh_to_xyxy(boxes1)
        box2 = self.convert_cxcywh_to_xyxy(boxes2)
        return box_iou(box1, box2)


def train_model(model, criterion, optimizer, train_loader, scheduler, num_epochs=10, model_name='PanelDetector'):
    try:
        model.load_state_dict(torch.load(f'{model_name}.pth'))
        print(f'Loaded {model_name}.pth')
    except FileNotFoundError:
        print(f'No existing model found. Training new {model_name} model.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    min_loss = 10000
    for epoch in range(num_epochs):
        model.train()

        start_time = time.time()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()

            running_loss += loss.item()

            if loss.item() <= min_loss:
                torch.save(model.state_dict(), f'{model_name}.pth')

            print(f'Epoch {epoch + 1}, Batch: {batch_idx + 1}, Loss: {loss.item()}')
            if batch_idx % 10 == 0:
                obj_mask = labels[..., 4] > 0
                noobj_mask = labels[..., 4] == 0

                # Compute losses for the bounding box coordinates (only for object-containing cells)
                box_predictions = outputs[..., :4][obj_mask]
                box_targets = labels[..., :4][obj_mask]
                print(f"box_targets: {(box_targets * 10000).round() / 10000}")
                print(f"box_predictions: {(box_predictions * 10000).round() / 10000}")

        running_loss /= len(train_loader)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f'Epoch {epoch + 1}, Loss: {running_loss}, Time: {elapsed_time:.2f}s')

        scheduler.step()

        print(f'Epoch {epoch + 1}, Loss: {running_loss}')

    print('Finished Training')


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print('Starting training')
    set_seed()

    # train_loader = DataLoader(Manga109Dataset('Test'), batch_size=1)

    root_dir = 'Manga109'
    dataset = Manga109Dataset(root_dir)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)


    logging.basicConfig(filename='logs/training.log', level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CustomYOLO().to(device).float()
    criterion = YOLOLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 7], gamma=0.1)
    train_model(model, criterion, optimizer, train_loader, scheduler, num_epochs=100, model_name='PanelDetector')

    return 0


if __name__ == '__main__':
    main()
