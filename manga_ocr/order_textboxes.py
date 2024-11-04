import cv2
import os

def crop_and_save_bboxes_by_class(manga_file_name):
    """
        !! CHANGE PATH !!
    """
    image_path = f"C:/Users/Zed/Desktop/CCNY Classes/2024 FALL/CSC 59867 Senior Project II/Project/Senior-Design/detected_panels/{manga_file_name}.jpg"
    annotation_path = f"C:/Users/Zed/Desktop/CCNY Classes/2024 FALL/CSC 59867 Senior Project II/Project/Senior-Design/detected_panels/annotation_sorted_textboxes.txt"
    
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    output_dir = "./detected_panels/sorted_text_boxes"
    os.makedirs(output_dir, exist_ok=True)  

    with open(annotation_path, "r") as file:
        annotations = file.readlines()

    for idx, annotation in enumerate(annotations):
        _, center_x, center_y, width, height = map(float, annotation.split())

        # convert YOLO format to bounding box coordinates
        x1 = int((center_x - width / 2) * image_width)
        y1 = int((center_y - height / 2) * image_height)
        x2 = int((center_x + width / 2) * image_width)
        y2 = int((center_y + height / 2) * image_height)

        # crop textbox region from the image
        cropped_textbox = image[y1:y2, x1:x2]

        # save cropped image: "1_ocr.jpg", "2_ocr.jpg", etc.
        output_path = os.path.join(output_dir, f"{idx+1}_ocr.jpg")
        cv2.imwrite(output_path, cropped_textbox)

        print(f"Saved {output_path}")

    print("Extraction complete!")

def main():
    manga_file_name = 'UnbalanceTokyo_061_right'
    crop_and_save_bboxes_by_class(manga_file_name)

if __name__ == "__main__":
    main()
