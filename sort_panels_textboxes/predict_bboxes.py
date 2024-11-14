from ultralytics import YOLO
from PIL import Image 
import numpy as np

def get_yolo_labels(pred_result, image_width, image_height):
    boxes = pred_result[0].boxes.xyxyn
    class_ids = pred_result[0].boxes.cls
    confidences = pred_result[0].boxes.conf

    yolo_bboxes = []

    for box, class_id in zip(boxes, class_ids):
        xcenter, ycenter, width, height = box.tolist()
        yolo_bbox = (int(class_id), (xcenter, ycenter, width, height))
        yolo_bboxes.append(yolo_bbox)

    return yolo_bboxes

def get_class_labels(yolo_labels):
    face_labels = []
    body_labels = []
    frame_labels = []
    text_labels = []

    for item in yolo_labels:
        if item[0] == 0:
            face_labels.append(item)
        elif item[0] == 1:
            body_labels.append(item)
        elif item[0] == 2:
            text_labels.append(item)
        elif item[0] == 3:
            frame_labels.append(item)
    
    return face_labels, body_labels, frame_labels, text_labels


