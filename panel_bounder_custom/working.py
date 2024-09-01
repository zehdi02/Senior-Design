import os
import time
import random
import logging

import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.ops import box_iou
from torchvision.transforms import Compose, Grayscale, ToTensor, Resize, Normalize
from torch.utils.tensorboard import SummaryWriter

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
        image = Image.open(image_path).convert('L')
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


class CustomYOLO(nn.Module):
    def __init__(self):
        super(CustomYOLO, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=1),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3),
            nn.Tanh(),
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=2, out_channels=3, kernel_size=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(2, stride=2),
            nn.AdaptiveAvgPool2d((448, 448)),
            nn.Sigmoid(),
        )

        self.conv1 = nn.Conv2d(3, 64, 7, stride=1, padding=3)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, 3)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.convs = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 512, 3, padding=1),
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.5)

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

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc1 = nn.Linear(8 * 8 * 1024, 4096)
        self.fc2 = nn.Linear(4096, 8 * 8 * 4) # 5 if confidence score is included

    def forward(self, x):
        x = self.reduce(x)
        assert not torch.isnan(x).any(), "reduce"

        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.dropout(self.pool3(F.leaky_relu(self.convs(x))))
        x = self.pool4(F.relu(self.convs2(x)))
        x = self.convs3(x)

        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)

        x = x.view(-1, 8, 8, 4) # 5 if conf
        x = torch.sigmoid(x)

        return x


class YOLOLoss(nn.Module):
    def __init__(self, l_center=2, l_dim=10, l_iou=10, l_noobj=0):
        super(YOLOLoss, self).__init__()
        self.l_center = l_center
        self.l_dim = l_dim
        self.l_noobj = l_noobj
        self.l_iou = l_iou

    def forward(self, predictions, target):
        # Masks where target boxes are present or absent
        obj_mask = target[..., 3] > 0 #  4 if conf
        # noobj_mask = target[..., 4] == 0

        # Compute losses for the bounding box coordinates (only for object-containing cells)
        box_predictions = predictions[..., :4][obj_mask]
        box_targets = target[..., :4][obj_mask]
        print(box_predictions[0], box_targets[0])

        # Calculate IoU loss
        iou_scores = self.calculate_iou(box_predictions, box_targets)
        iou_loss = 1 - iou_scores.mean()  # Average IoU loss across all positive samples

        # MSE loss for bounding box coordinates
        box_loss = F.mse_loss(torch.sqrt(box_predictions[..., 2:4]), torch.sqrt(box_targets[..., 2:4]), reduction='sum')
        center_loss = F.mse_loss(box_predictions[..., :2], box_targets[..., :2], reduction='sum')

        # # Compute the loss for confidence scores
        # obj_confidence_loss = F.mse_loss(predictions[..., 4][obj_mask], target[..., 4][obj_mask], reduction='sum')
        # noobj_confidence_loss = F.mse_loss(predictions[..., 4][noobj_mask], target[..., 4][noobj_mask], reduction='sum')

        # Calculate the total loss incorporating the differential weighting
        total_loss = (
                (self.l_center * center_loss) +
                (self.l_dim * box_loss) +
                (self.l_iou * iou_loss) ) / predictions.size(0) #+
                # obj_confidence_loss + (self.l_noobj * noobj_confidence_loss)) \ predictions.size(0)

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


def train_model(model, criterion, optimizer, train_loader, num_epochs=10, patience=16, model_name='PanelDetector', writer=None):
    try:
        model.load_state_dict(torch.load(f'models/{model_name}.pth'))
        print(f'Loaded {model_name}.pth')
    except FileNotFoundError:
        print(f'No existing model found. Training new {model_name} model.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    best_loss = float('inf')
    batches_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2, norm_type=1)
            optimizer.step()

            running_loss += loss.item()

            if loss.item() < best_loss:
                best_loss = loss.item()
                batches_without_improvement = 0
                torch.save(model.state_dict(), f'models/{model_name}.pth')
            else:
                batches_without_improvement += 1

            logging.info(f'Epoch {epoch + 1}, Batch: {batch_idx + 1}, Loss: {loss.item()}')
            print(f'Epoch {epoch + 1}, Batch: {batch_idx + 1}, Loss: {loss.item()}')
            if writer:
                writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + batch_idx)

            if batches_without_improvement >= patience:
                print('Early stopping due to loss stagnation')
                if writer:
                    writer.close()
                return

        running_loss /= len(train_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time

        logging.info(f'Epoch {epoch + 1}, Loss: {running_loss}, Time: {elapsed_time:.2f}s')
        print(f'Epoch {epoch + 1}, Loss: {running_loss}, Time: {elapsed_time:.2f}s')

    if writer:
        writer.close()


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print('Starting')
    set_seed(99)

    root_dir = '../Manga109'
    dataset = Manga109Dataset(root_dir)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    model_name = 'PanelDetector_v2.1'
    logging.basicConfig(filename=f'logs/{model_name}.log', level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CustomYOLO().to(device).float()

    writer = SummaryWriter(f'runs/{model_name}')
    writer.add_graph(model, torch.rand(1, 1, 1024, 1024).to(device))
    writer.close()

    criterion = YOLOLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_model(model, criterion, optimizer, train_loader, num_epochs=1, model_name=model_name, writer=writer)

    return 0


if __name__ == '__main__':
    main()
