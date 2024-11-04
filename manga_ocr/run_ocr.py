import os
from manga_ocr import MangaOcr

mocr = MangaOcr()

sorted_textbox_dir = "./detected_panels/sorted_text_boxes"
current_dir = os.path.dirname(os.path.abspath(__file__))

transcript_file = os.path.join(current_dir, "transcript_MangaVision.txt")

with open(transcript_file, "w") as transcript:
    for idx, image_file in enumerate(sorted(os.listdir(sorted_textbox_dir)), start=1):
        image_path = os.path.join(sorted_textbox_dir, image_file)

        extracted_text = mocr(image_path)

        transcript.write(f"{idx} character_name: {extracted_text}\n")

        print(f"Processed {image_file} and extracted text.")

print("OCR extraction complete! Transcript saved to transcript_MangaVision.txt.")
