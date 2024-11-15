import cv2
import os
from sort_panel_textboxes import *

def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    """Draw bounding box with label on the image."""
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 6, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA) 
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, lw / 6, txt_color, thickness=tf, lineType=cv2.LINE_AA)

def draw_sorted_bounding_boxes(image_path, panels_list, text_boxes_list, panels_conf, text_boxes_conf):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    height, width, _ = image.shape

    def yolo_to_bbox(yolo_label):
        class_id, x_center, y_center, w, h = yolo_label
        x_center *= width
        y_center *= height
        w *= width
        h *= height
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
        return class_id, x1, y1, x2, y2

    panel_color = (0, 0, 255)
    text_color = (0, 255, 0)
    panel_circle_color = (0, 0, 155)
    text_circle_color = (0, 155, 0)

    def plot_bboxes(image, boxes, confidences, label_name, color, circle_color):
        centers = []
        for i, (box, conf) in enumerate(zip(boxes, confidences), start=1):
            class_id, x1, y1, x2, y2 = yolo_to_bbox(box)
            bbox = [x1, y1, x2, y2]
            label = f"{label_name}_{i} ({conf:.2f})" 

            # draw box with label
            box_label(image, bbox, label=label, color=color)

            # calculate center and draw circle
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            centers.append((center_x, center_y))
            cv2.circle(image, (center_x, center_y), 5, circle_color, -1)

        # draw connecting arrows at center of bounding boxes
        for j in range(len(centers) - 1):
            cv2.arrowedLine(image, centers[j], centers[j + 1], color, 2, tipLength=0.05)

    # draw bounding boxes for panels and text boxes
    plot_bboxes(image, panels_list, panels_conf, "panel", panel_color, panel_circle_color)
    plot_bboxes(image, text_boxes_list, text_boxes_conf, "text", text_color, text_circle_color)

    output_dir = "images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("predicted_bboxes") and f.endswith(".jpg")]
    next_index = len(existing_files) + 1
    output_path = os.path.join(output_dir, f"predicted_bboxes_{next_index}.jpg")

    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

    # cv2.imshow('MangaVision', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image
