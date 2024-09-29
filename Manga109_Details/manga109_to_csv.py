import os
import csv
import xml.etree.ElementTree as ET


def parse_xml_to_csv(xml_folder, csv_file):
    # Create the CSV file if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['book', 'page_index', 'type', 'id', 'box_xmin', 'box_ymin',
                             'box_xmax', 'box_ymax', 'x_center', 'y_center', 'area', 'frame_id', 'text'])

    rows = []
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_folder, xml_file))
            root = tree.getroot()
            book_title = root.get('title')
            print(f"Processing book: {book_title}")

            for page in root.findall('pages/page'):
                page_index = page.get('index')

                # Collect all frame bounding boxes and write rows for them
                frames = []
                for frame in page.findall('frame'):
                    frame_id = frame.get('id')
                    frame_xmin = int(frame.get('xmin'))
                    frame_ymin = int(frame.get('ymin'))
                    frame_xmax = int(frame.get('xmax'))
                    frame_ymax = int(frame.get('ymax'))
                    x_center = (frame_xmin + frame_xmax) / 2
                    y_center = (frame_ymin + frame_ymax) / 2
                    area = (frame_xmax - frame_xmin) * (frame_ymax - frame_ymin)
                    frames.append((frame_id, frame_xmin, frame_ymin, frame_xmax, frame_ymax))
                    rows.append([book_title, page_index, 'frame', frame_id, frame_xmin, frame_ymin,
                                 frame_xmax, frame_ymax, x_center, y_center, area, None, None])

                for annotation_type in ['text', 'face', 'body']:
                    for annotation in page.findall(annotation_type):
                        annotation_id = annotation.get('id')
                        xmin = int(annotation.get('xmin'))
                        ymin = int(annotation.get('ymin'))
                        xmax = int(annotation.get('xmax'))
                        ymax = int(annotation.get('ymax'))
                        x_center = (xmin + xmax) / 2
                        y_center = (ymin + ymax) / 2
                        area = (xmax - xmin) * (ymax - ymin)
                        text = annotation.text if annotation_type == 'text' else None

                        # Determine which frame the center is in
                        frame_id = next((frame_id for frame_id, frame_xmin, frame_ymin, frame_xmax, frame_ymax in frames
                                         if frame_xmin <= x_center <= frame_xmax and frame_ymin <= y_center <= frame_ymax), None)

                        rows.append([book_title, page_index, annotation_type, annotation_id, xmin, ymin,
                                     xmax, ymax, x_center, y_center, area, frame_id, text])

    # Write all rows to the CSV file at once
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


if __name__ == "__main__":
    xml_folder = '../Manga109/annotations'
    csv_file = '../Manga109/Manga109.csv'
    parse_xml_to_csv(xml_folder, csv_file)