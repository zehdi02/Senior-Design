import cv2
from sort_panel_textboxes import *

def draw_bounding_boxes(image_path, panels_list, text_boxes_list):
    # load  image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    height, width, _ = image.shape
    # convert YOLO format to OpenCV rectangle format
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

    # define colors: RED for panels; GREEN for text boxes
    panel_color = (0, 0, 255)  
    panel_arrow_color = (0, 0, 255)   
    panel_circle_color = (0, 0, 155)  
    text_color = (0, 255, 0)  
    text_arrow_color = (0, 255, 0)   
    text_circle_color = (0, 155, 0)

    # draw bounding boxes for panels
    panel_centers = []
    for i, panel in enumerate(panels_list, start=1): 
        class_id, x1, y1, x2, y2 = yolo_to_bbox(panel)
        cv2.rectangle(image, (x1, y1), (x2, y2), panel_color, 2)
        cv2.putText(image, f"panel", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, .8, panel_color, 2)

        # panel position order at inner top-right of panel bbox
        position_text = str(i)
        cv2.putText(image, position_text, (x2-20, y1+20),  
                    cv2.FONT_HERSHEY_DUPLEX, .8, panel_color, 2)

        # calc center of panel
        center_x, center_y = ((x1 + x2) // 2, (y1 + y2) // 2)
        panel_centers.append((center_x, center_y))

        # draw circle at center 
        cv2.circle(image, (center_x, center_y), 5, panel_circle_color, -1)

    # draw arrows for panel list order
    for i in range(len(panel_centers) - 1):
        cv2.arrowedLine(image, panel_centers[i], panel_centers[i+1], panel_arrow_color, 2, tipLength=0.05)

    # draw bounding boxes for text boxes
    text_box_centers = []
    for i, text_box in enumerate(text_boxes_list, start=1):  
        class_id, x1, y1, x2, y2 = yolo_to_bbox(text_box)
        cv2.rectangle(image, (x1, y1), (x2, y2), text_color, 2)
        cv2.putText(image, f"text", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, .5, text_color, 2)

        # text box position order at inner top-right of panel bbox
        position_text = str(i)
        cv2.putText(image, position_text, (x2-20, y1+20),  
                    cv2.FONT_HERSHEY_DUPLEX, .5, text_color, 2)

        # calc center of text box
        center_x, center_y = ((x1 + x2) // 2, (y1 + y2) // 2)
        text_box_centers.append((center_x, center_y))

        # draw circle at center 
        cv2.circle(image, (center_x, center_y), 5, text_circle_color, -1)

    # draw arrows for text box list order
    for i in range(len(text_box_centers) - 1):
        cv2.arrowedLine(image, text_box_centers[i], text_box_centers[i+1], text_arrow_color, 2, tipLength=0.05)

    # create dir "images" if it doesn't exist
    output_dir = "images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("predicted_bboxes") and f.endswith(".jpg")]
    next_index = len(existing_files) + 1
    output_path = os.path.join(output_dir, f"predicted_bboxes_{next_index}.jpg")

    # save image
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")

    # display image
    cv2.imshow('MangaVision', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# img_fp = './032.png'
img_fp = './watchmen.jpg'

sorted_panels_list, sorted_text_boxes_list = sorting_pipeline(img_fp)

draw_bounding_boxes(img_fp, sorted_panels_list, sorted_text_boxes_list)