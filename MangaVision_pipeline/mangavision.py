import os
import cv2
from PIL import Image
from io import BytesIO
from manga_ocr import MangaOcr

from sort_panel_textboxes import *
from display_sorted_panels_textboxes import *

def generate_transcript(img_fp, sorted_text_boxes_list):
    """
    1. crops the sorted text boxes from an image.
    2. apply OCR on each sorted cropped region.
    3. saves the extracted texts in 'transcripts' directory.
    """
    image = cv2.imread(img_fp)
    if image is None:
        print(f"Error: Unable to load image from {img_fp}")
        return
    
    image_height, image_width = image.shape[:2]
    
    mocr = MangaOcr()
    
    transcript_dir = "./transcripts"
    os.makedirs(transcript_dir, exist_ok=True)
    
    existing_files = [f for f in os.listdir(transcript_dir) if f.startswith("ocr_transcript_") and f.endswith(".txt")]
    if existing_files:
        existing_numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
        next_index = max(existing_numbers) + 1
    else:
        next_index = 1

    output_txt_file = os.path.join(transcript_dir, f"ocr_transcript_{next_index}.txt")

    with open(output_txt_file, 'w', encoding='utf-8') as file:
        for idx, annotation in enumerate(sorted_text_boxes_list):
            class_id, center_x, center_y, width, height = map(float, annotation)
            
            # convert YOLO format to xyxy
            x1 = int((center_x - width / 2) * image_width) - 5
            y1 = int((center_y - height / 2) * image_height) - 5
            x2 = int((center_x + width / 2) * image_width) + 5
            y2 = int((center_y + height / 2) * image_height) + 5

            # crop textbox regions from the whole image
            cropped_textbox = image[y1:y2, x1:x2]
            
            # convert OpenCV image (NumPy array) to PIL Image directly
            img_pil = Image.fromarray(cv2.cvtColor(cropped_textbox, cv2.COLOR_BGR2RGB))

            # perform OCR
            text = mocr(img_pil)
            
            file.write(f"Textbox {idx+1}: {text}\n")

    print(f"OCR extraction completed!\nTranscript saved to {output_txt_file}")

img_fp = '155.png'

sorted_text_boxes_list, sorted_panels_list, sorted_text_boxes_conf_list, sorted_panels_conf_list = sorting_pipeline(img_fp)

generate_transcript(img_fp, sorted_text_boxes_list)
draw_bounding_boxes(img_fp, sorted_panels_list, sorted_text_boxes_list, sorted_panels_conf_list, sorted_text_boxes_conf_list)
