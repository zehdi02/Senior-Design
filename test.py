import numpy as np
import torch
from PIL import Image
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt, patches
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Grayscale, ToTensor, Resize, Normalize, ToPILImage
from working import Manga109Dataset, CustomYOLO


def get_by_path(title, page):
    annotation_path = f'Manga109/annotations/{title}.xml'
    image_path = f'Manga109/images/{title}/{int(page):03}.jpg'

    image = Image.open(image_path).convert('L')
    transform = Compose([
        Resize((1024, 1024)),
        Grayscale(),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(image)
    image = ToPILImage()(image_tensor)


    bboxes = []
    y_true = torch.zeros((8, 8, 5))
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

        # Calculate grid cell indices for the center
        center_col = int(x_center * 8)
        center_row = int(y_center * 8)

        y_true[center_col, center_row, :] = torch.tensor([x_center, y_center, width, height, 1])

    return image, image_tensor, bboxes, y_true

def predict(model, image_tensor):
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
    return output

def show_image_colormap(image, bboxes, target, pred):
    # get mask for pred based on ytrue centers only
    obj_mask = target[..., 4] == 1
    print(target[obj_mask])
    pred = pred[..., :4][obj_mask]
    print(pred)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')  # Display the grayscale image

    # Set up color map for bounding boxes
    cmap = plt.get_cmap('tab20', len(bboxes))  # Generate a color map with enough colors

    # Draw each frame as a rectangle for ground truth and prediction
    for idx, (gt_bbox, pr_bbox) in enumerate(zip(bboxes, pred)):
        # Denormalize the bounding box coordinates
        pr_bbox = [coord * 1024 for coord in pr_bbox.cpu()]

        # Predicted bbox
        pr_rect = patches.Rectangle((pr_bbox[0] - pr_bbox[2] / 2, pr_bbox[1] - pr_bbox[3] / 2), pr_bbox[2], pr_bbox[3],
                                    linewidth=1, edgecolor=cmap(idx), linestyle='-', facecolor='none')
        ax.add_patch(pr_rect)

    plt.show()

def visual_eval(model, title, page):
    image, image_tensor, bboxes, y_true = get_by_path(title, page)
    pred = predict(model, image_tensor.unsqueeze(0)).squeeze(0)
    show_image_colormap(image, bboxes, y_true, pred)

def main():
    # root_dir = 'Manga109'
    # dataset = Manga109Dataset(root_dir)
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    #
    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CustomYOLO().to(device).float()

    x = torch.randn(1, 1, 1024, 1024).to(device)
    make_dot

    for i in range(3,4):
        model.load_state_dict(torch.load(f'models/PanelDetector_v{i+1}.pth'))

        visual_eval(model, 'ARMS', 11)
        visual_eval(model, 'Hamlet', 11)


if __name__ == '__main__':
    main()
