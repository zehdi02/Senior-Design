import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from manga_ocr import MangaOcr

from sort_panel_textboxes import *
from display_sorted_panels_textboxes import *

def convert_image_bytes_to_cv2(image_bytes):
    """
    Convert image bytes to an OpenCV image.
    """
    # Check if the input is indeed bytes
    if isinstance(image_bytes, bytes):
        # Convert bytes data to a NumPy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        # Decode the NumPy array to get an OpenCV image
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Error decoding the image bytes.")
        return img
    else:
        raise TypeError("Input data is not in bytes format.")

def yolo_to_xyxy(center_x, center_y, width, height, image_height, image_width):
    x1 = int((center_x - width / 2) * image_width) - 5
    y1 = int((center_y - height / 2) * image_height) - 5
    x2 = int((center_x + width / 2) * image_width) + 5
    y2 = int((center_y + height / 2) * image_height) + 5

    return x1, y1, x2, y2

def generate_transcript(img_fp, sorted_text_boxes_list, is_api = False, mocr = None):
    """
    1. crops the sorted text boxes from an image.
    2. apply OCR on each sorted cropped region.
    3. saves the extracted texts in 'transcripts' directory.
    """

    is_api = 0
    if isinstance(img_fp, bytes):
        print('Image is in bytes. Now converting to cv2.')
        is_api = 1
        image = convert_image_bytes_to_cv2(img_fp)
    else:
        mocr = MangaOcr()
        image = cv2.imread(img_fp)
        if image is None:
            print(f"Error: Unable to load image from {img_fp}")
            return

    image_height, image_width = image.shape[:2]
    
    if is_api == 0:
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
                x1, y1, x2, y2 = yolo_to_xyxy(center_x, center_y, width, height, image_height, image_width)

                # crop textbox regions from the whole image
                cropped_textbox = image[y1:y2, x1:x2]
                
                # convert OpenCV image (NumPy array) to PIL Image directly
                img_pil = Image.fromarray(cv2.cvtColor(cropped_textbox, cv2.COLOR_BGR2RGB))

                # perform OCR
                text = mocr(img_pil)
                
                file.write(f"Textbox {idx+1}: {text}\n")
        print(f"OCR extraction completed!\nTranscript saved to {output_txt_file}")
    else:
        extracted_texts = []  
        for idx, annotation in enumerate(sorted_text_boxes_list):
            class_id, center_x, center_y, width, height = map(float, annotation)

            x1, y1, x2, y2 = yolo_to_xyxy(center_x, center_y, width, height, image_height, image_width)

            cropped_textbox = image[y1:y2, x1:x2]
            
            img_pil = Image.fromarray(cv2.cvtColor(cropped_textbox, cv2.COLOR_BGR2RGB))

            text = mocr(img_pil)
            
            extracted_texts.append(text)
        print(f"OCR extraction completed!")

    return extracted_texts

# img_fp = '../MangaVision_API/sample_data/manga_page.png'

# sorted_text_boxes_list, sorted_panels_list, sorted_text_boxes_conf_list, sorted_panels_conf_list = sorting_pipeline(img_fp)

# generate_transcript(img_fp, sorted_text_boxes_list)
# draw_sorted_bounding_boxes(img_fp, sorted_panels_list, sorted_text_boxes_list, sorted_panels_conf_list, sorted_text_boxes_conf_list)
