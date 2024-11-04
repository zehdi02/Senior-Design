import cv2
import os

# function to crop and save bounding boxes & annotations for panels and text boxes
def crop_and_save_bboxes_by_class(manga_file_name):
    """
        !! CHANGE PATH !!
    """
    full_image_path = f"C:/Users/Zed/Desktop/CCNY Classes/2024 FALL/CSC 59867 Senior Project II/Project/Senior-Design/Manga109_YOLO/train/images/{manga_file_name}.jpg"
    full_label_path = f"C:/Users/Zed/Desktop/CCNY Classes/2024 FALL/CSC 59867 Senior Project II/Project/Senior-Design/Manga109_YOLO/train/labels/{manga_file_name}.txt"
    
    main_output_dir = "./detected_panels"
    panels_dir = os.path.join(main_output_dir, "panels")
    text_boxes_dir = os.path.join(main_output_dir, "text_boxes")
    os.makedirs(panels_dir, exist_ok=True)
    os.makedirs(text_boxes_dir, exist_ok=True)

    # Read the image
    img = cv2.imread(full_image_path)
    if img is None:
        print(f"Error loading image: {full_image_path}")
        return

    base_name = os.path.splitext(os.path.basename(full_image_path))[0]

    panels_annotation_path = os.path.join(panels_dir, f"{base_name}_panels_annotations.txt")
    text_boxes_annotation_path = os.path.join(text_boxes_dir, f"{base_name}_textboxes_annotations.txt")
    
    with open(full_label_path, 'r') as label_file, \
         open(panels_annotation_path, 'w') as panels_file, \
         open(text_boxes_annotation_path, 'w') as text_boxes_file:
        
        for line in label_file.readlines():
            data = line.strip().split()
            class_id = int(data[0])
            x_center = float(data[1]) * img.shape[1]
            y_center = float(data[2]) * img.shape[0]
            bbox_width = float(data[3]) * img.shape[1]
            bbox_height = float(data[4]) * img.shape[0]

            # calculate bounding box
            x_min = int(x_center - bbox_width / 2)
            y_min = int(y_center - bbox_height / 2)
            x_max = int(x_center + bbox_width / 2)
            y_max = int(y_center + bbox_height / 2)

            # crop and save for panels
            if class_id == 3:
                cropped_img = img[y_min:y_max, x_min:x_max]
                output_image_path = os.path.join(panels_dir, f"{base_name}_panel_{x_min}_{y_min}.jpg")
                cv2.imwrite(output_image_path, cropped_img)
                panels_file.write(f"{class_id} {data[1]} {data[2]} {data[3]} {data[4]}\n")
            
            # crop and save for text boxes 
            elif class_id == 2:
                cropped_img = img[y_min:y_max, x_min:x_max]
                output_image_path = os.path.join(text_boxes_dir, f"{base_name}_textbox_{x_min}_{y_min}.jpg")
                cv2.imwrite(output_image_path, cropped_img)
                text_boxes_file.write(f"{class_id} {data[1]} {data[2]} {data[3]} {data[4]}\n")

    print(f"Cropped images and annotations saved at {full_image_path}.")

def main():
    manga_file_name = 'UnbalanceTokyo_061_right'
    crop_and_save_bboxes_by_class(manga_file_name)

if __name__ == "__main__":
    main()
