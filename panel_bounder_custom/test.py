import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import torch.nn.functional as F
from matplotlib import pyplot as plt, patches
from torchvision.transforms import Compose, Grayscale, ToTensor, Resize, Normalize, ToPILImage
from working import CustomYOLO

def get_by_path(title, page, device):
    annotation_path = f'Manga109/annotations/{title}.xml'
    image_path = f'Manga109/images/{title}/{int(page):03}.jpg'

    image = Image.open(image_path).convert('L')
    transform = Compose([
        Resize((1024, 1024)),
        Grayscale(),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(image).to(device)
    image = ToPILImage()(image_tensor.cpu())

    bboxes = []
    y_true = torch.zeros((8, 8, 5)).to(device)
    frames = ET.parse(annotation_path).getroot().find(f".//page[@index='{str(page)}']").findall('frame')
    for frame in frames:
        bbox = (
            int(frame.get('xmin')) / 1654,
            int(frame.get('ymin')) / 1170,
            int(frame.get('xmax')) / 1654,
            int(frame.get('ymax')) / 1170,
        )
        bboxes.append(bbox)

        xmin, ymin, xmax, ymax = bbox

        x_center = ((xmin + xmax) / 2)
        y_center = ((ymin + ymax) / 2)
        width = (xmax - xmin)
        height = (ymax - ymin)

        center_col = int(x_center * 8)
        center_row = int(y_center * 8)

        y_true[center_col, center_row, :] = torch.tensor([x_center, y_center, width, height, 1], device=device)

    return image, image_tensor, bboxes, y_true

def predict(model, image_tensor):
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
    return output

def show_image_colormap(image, bboxes, target, pred):
    obj_mask = target[..., 4] == 1
    pred = pred[..., :4][obj_mask]

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    cmap = plt.get_cmap('tab20', len(bboxes))

    for idx, (gt_bbox, pr_bbox) in enumerate(zip(bboxes, pred)):
        pr_bbox = [coord * 1024 for coord in pr_bbox.cpu()]

        pr_rect = patches.Rectangle((pr_bbox[0] - pr_bbox[2] / 2, pr_bbox[1] - pr_bbox[3] / 2), pr_bbox[2], pr_bbox[3],
                                    linewidth=3, edgecolor=cmap(idx), linestyle='-', facecolor='none')
        ax.add_patch(pr_rect)

    plt.show()

def visual_eval(model, title, page, device):
    image, image_tensor, bboxes, y_true = get_by_path(title, page, device)
    pred = predict(model, image_tensor.unsqueeze(0)).squeeze(0)
    show_image_colormap(image, bboxes, y_true, pred)

def calculate_metrics(y_true, y_pred, iou_threshold=0.5):
    device = y_true.device
    obj_mask = y_true[..., 4] == 1
    y_true = y_true[obj_mask].to(device)
    y_pred = y_pred[obj_mask].to(device)

    x_true, y_true, w_true, h_true = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
    x_pred, y_pred, w_pred, h_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]

    def calculate_iou(x1, y1, w1, h1, x2, y2, w2, h2):
        x1_min, y1_min, x1_max, y1_max = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
        x2_min, y2_min, x2_max, y2_max = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

        inter_min_x = torch.max(x1_min, x2_min)
        inter_min_y = torch.max(y1_min, y2_min)
        inter_max_x = torch.min(x1_max, x2_max)
        inter_max_y = torch.min(y1_max, y2_max)

        inter_area = torch.clamp(inter_max_x - inter_min_x, min=0) * torch.clamp(inter_max_y - inter_min_y, min=0)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area
        return iou

    iou = calculate_iou(x_true, y_true, w_true, h_true, x_pred, y_pred, w_pred, h_pred)
    iou_loss = 1 - iou.mean()

    center_loss = F.mse_loss(torch.stack([x_true, y_true], dim=-1), torch.stack([x_pred, y_pred], dim=-1))
    wh_loss = F.mse_loss(torch.stack([w_true, h_true], dim=-1), torch.stack([w_pred, h_pred], dim=-1))

    precision = (iou > iou_threshold).float().mean()
    recall = (iou > iou_threshold).float().sum() / y_true.size(0)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    return iou_loss.item(), center_loss.item(), wh_loss.item(), precision.item(), recall.item(), f1_score.item()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CustomYOLO().to(device).float()

    model.load_state_dict(torch.load(f'models/PanelDetector_v2.1.pth', map_location=device))

    total_iou_loss = 0
    total_center_loss = 0
    total_wh_loss = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    num_samples = 0

    titles = os.listdir('../Manga109/images')
    titles = titles[:10] # For testing purposes, only use the first 10 title
    for title in titles:
        print(title)
        for page in os.listdir(f'Manga109/images/{title}'):
            page = int(page.split('.')[0])
            image, image_tensor, bboxes, y_true = get_by_path(title, page, device)
            if not bboxes: continue
            y_pred = predict(model, image_tensor.unsqueeze(0)).squeeze(0)
            iou_loss, center_loss, wh_loss, precision, recall, f1_score = calculate_metrics(y_true, y_pred)

            total_iou_loss += iou_loss
            total_center_loss += center_loss
            total_wh_loss += wh_loss
            total_precision += precision
            total_recall += recall
            total_f1_score += f1_score
            num_samples += 1

    avg_iou_loss = total_iou_loss / num_samples
    avg_center_loss = total_center_loss / num_samples
    avg_wh_loss = total_wh_loss / num_samples
    avg_precision = total_precision / num_samples
    avg_recall = total_recall / num_samples
    avg_f1_score = total_f1_score / num_samples

    print(f'Average IoU Loss: {avg_iou_loss}')
    print(f'Average Center Loss: {avg_center_loss}')
    print(f'Average Width-Height Loss: {avg_wh_loss}')
    print(f'Average Precision: {avg_precision}')
    print(f'Average Recall: {avg_recall}')
    print(f'Average F1 Score: {avg_f1_score}')

if __name__ == '__main__':
    main()
