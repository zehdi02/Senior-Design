import os
import json
import logging
import manga109api
import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load annotations for all books
def load_annotations(parser, books):
    logging.info("Loading annotations for books...")
    annotations = {}
    for book in books:
        logging.info(f"Loading annotations for book: {book}")
        annotations[book] = parser.get_annotation(book=book)
    logging.info("Finished loading annotations.")
    return annotations

# Save annotations to a JSON file
def save_annotations_to_file(annotations, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)

# Load annotations from a JSON file
def load_annotations_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calc_avg_median_std_dev(data):
    avg = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    return avg, median, std_dev


# Dataset statistics: total number of books, images, characters, and bounding boxes for frames, faces, bodies, texts
def dataset_statistics(annotations):
    num_books = len(annotations)
    num_images = 0
    num_characters = 0
    num_frames = 0
    num_faces = 0
    num_bodies = 0
    num_texts = 0

    for book, annotation in annotations.items():
        num_images += len(annotation['page'])
        num_characters += len(annotation['character'])
        for page in annotation['page']:
            num_frames += len(page.get('frame', []))
            num_faces += len(page.get('face', []))
            num_bodies += len(page.get('body', []))
            num_texts += len(page.get('text', []))

    print(f'Total number of books: {num_books}')
    print(f'Total number of images: {num_images}')
    print(f'Total number of characters: {num_characters}')
    print(f'Total number of bounding boxes for frames: {num_frames}')
    print(f'Total number of bounding boxes for texts: {num_texts}')
    print(f'Total number of bounding boxes for faces: {num_faces}')
    print(f'Total number of bounding boxes for bodies: {num_bodies}')


# show number of pages/images per book in descending order with average/median/standard deviation
def show_images_per_book(annotations):
    images_per_book = []

    for annotation in annotations.values():
        num_images = len(annotation['page'])
        images_per_book.append(num_images)

    # Calculate statistics
    images_per_book = np.array(images_per_book)
    avg, median, std_dev = calc_avg_median_std_dev(images_per_book)

    # Plot bar graph descending order and show average, median, and standard deviation on the graph as horizontal dotted lines
    sorted_indices = np.argsort(images_per_book)[::-1]
    sorted_images_per_book = images_per_book[sorted_indices]
    book_titles = list(annotations.keys())
    sorted_book_titles = [book_titles[i] for i in sorted_indices]

    plt.figure(figsize=(10, 5))
    plt.bar(sorted_book_titles, sorted_images_per_book)
    plt.xticks(rotation=90)
    plt.title('Number of Images per Book')
    plt.xlabel('Book')
    plt.ylabel('Number of Images')
    plt.grid(axis='y')
    plt.axhline(y=avg, color='r', linestyle='--', linewidth=2, label=f'Average: {avg:.2f}')
    plt.axhline(y=median, color='g', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
    plt.axhline(y=std_dev, color='b', linestyle='--', linewidth=2, label=f'Std Dev: {std_dev:.2f}')
    plt.legend()
    plt.show()

    print(f'Average number of images per book: {avg}')
    print(f'Median number of images per book: {median}')
    print(f'Standard deviation of images per book: {std_dev}')

# Show number of characters per book, with average/median/standard deviation
def show_characters_per_book(annotations):
    characters_per_book = []

    for annotation in annotations.values():
        num_characters = len(annotation['character'])
        characters_per_book.append(num_characters)

    # Calculate statistics
    characters_per_book = np.array(characters_per_book)
    avg, median, std_dev = calc_avg_median_std_dev(characters_per_book)

    # Plot bar graph descending order and show average, median, and standard deviation on the graph as horizontal dotted lines
    sorted_indices = np.argsort(characters_per_book)[::-1]
    sorted_characters_per_book = characters_per_book[sorted_indices]
    book_titles = list(annotations.keys())
    sorted_book_titles = [book_titles[i] for i in sorted_indices]

    plt.figure(figsize=(10, 5))
    plt.bar(sorted_book_titles, sorted_characters_per_book)
    plt.xticks(rotation=90)
    plt.title('Number of Characters per Book')
    plt.xlabel('Book')
    plt.ylabel('Number of Characters')
    plt.grid(axis='y')
    plt.axhline(y=avg, color='r', linestyle='--', linewidth=2, label=f'Average: {avg:.2f}')
    plt.axhline(y=median, color='g', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
    plt.axhline(y=std_dev, color='b', linestyle='--', linewidth=2, label=f'Std Dev: {std_dev:.2f}')
    plt.legend()
    plt.show()

    print(f'Average number of characters per book: {avg}')
    print(f'Median number of characters per book: {median}')
    print(f'Standard deviation of characters per book: {std_dev}')

# Show number of frames per book, with average/median/standard deviation
def show_frames_per_book(annotations):
    frames_per_book = []

    for annotation in annotations.values():
        num_frames = sum([len(page.get('frame', [])) for page in annotation['page']])
        frames_per_book.append(num_frames)

    # Calculate statistics
    frames_per_book = np.array(frames_per_book)
    avg, median, std_dev = calc_avg_median_std_dev(frames_per_book)

    # Plot bar graph descending order and show average, median, and standard deviation on the graph as horizontal dotted lines
    sorted_indices = np.argsort(frames_per_book)[::-1]
    sorted_frames_per_book = frames_per_book[sorted_indices]
    book_titles = list(annotations.keys())
    sorted_book_titles = [book_titles[i] for i in sorted_indices]

    plt.figure(figsize=(10, 5))
    plt.bar(sorted_book_titles, sorted_frames_per_book)
    plt.xticks(rotation=90)
    plt.title('Number of Frames per Book')
    plt.xlabel('Book')
    plt.ylabel('Number of Frames')
    plt.grid(axis='y')
    plt.axhline(y=avg, color='r', linestyle='--', linewidth=2, label=f'Average: {avg:.2f}')
    plt.axhline(y=median, color='g', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
    plt.axhline(y=std_dev, color='b', linestyle='--', linewidth=2, label=f'Std Dev: {std_dev:.2f}')
    plt.legend()
    plt.show()

    print(f'Average number of frames per book: {avg}')
    print(f'Median number of frames per book: {median}')
    print(f'Standard deviation of frames per book: {std_dev}')

# Show number of faces per book, with average/median/standard deviation
def show_faces_per_book(annotations):
    faces_per_book = []

    for annotation in annotations.values():
        num_faces = sum([len(page.get('face', [])) for page in annotation['page']])
        faces_per_book.append(num_faces)

    # Calculate statistics
    faces_per_book = np.array(faces_per_book)
    avg, median, std_dev = calc_avg_median_std_dev(faces_per_book)

    # Plot bar graph descending order and show average, median, and standard deviation on the graph as horizontal dotted lines
    sorted_indices = np.argsort(faces_per_book)[::-1]
    sorted_faces_per_book = faces_per_book[sorted_indices]
    book_titles = list(annotations.keys())
    sorted_book_titles = [book_titles[i] for i in sorted_indices]

    plt.figure(figsize=(10, 5))
    plt.bar(sorted_book_titles, sorted_faces_per_book)
    plt.xticks(rotation=90)
    plt.title('Number of Faces per Book')
    plt.xlabel('Book')
    plt.ylabel('Number of Faces')
    plt.grid(axis='y')
    plt.axhline(y=avg, color='r', linestyle='--', linewidth=2, label=f'Average: {avg:.2f}')
    plt.axhline(y=median, color='g', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
    plt.axhline(y=std_dev, color='b', linestyle='--', linewidth=2, label=f'Std Dev: {std_dev:.2f}')
    plt.legend()
    plt.show()

    print(f'Average number of faces per book: {avg}')
    print(f'Median number of faces per book: {median}')
    print(f'Standard deviation of faces per book: {std_dev}')

# Show number of bodies per book, with average/median/standard deviation
def show_bodies_per_book(annotations):
    bodies_per_book = []

    for annotation in annotations.values():
        num_bodies = sum([len(page.get('body', [])) for page in annotation['page']])
        bodies_per_book.append(num_bodies)

    # Calculate statistics
    bodies_per_book = np.array(bodies_per_book)
    avg, median, std_dev = calc_avg_median_std_dev(bodies_per_book)

    # Plot bar graph descending order and show average, median, and standard deviation on the graph as horizontal dotted lines
    sorted_indices = np.argsort(bodies_per_book)[::-1]
    sorted_bodies_per_book = bodies_per_book[sorted_indices]
    book_titles = list(annotations.keys())
    sorted_book_titles = [book_titles[i] for i in sorted_indices]

    plt.figure(figsize=(10, 5))
    plt.bar(sorted_book_titles, sorted_bodies_per_book)
    plt.xticks(rotation=90)
    plt.title('Number of Bodies per Book')
    plt.xlabel('Book')
    plt.ylabel('Number of Bodies')
    plt.grid(axis='y')
    plt.axhline(y=avg, color='r', linestyle='--', linewidth=2, label=f'Average: {avg:.2f}')
    plt.axhline(y=median, color='g', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
    plt.axhline(y=std_dev, color='b', linestyle='--', linewidth=2, label=f'Std Dev: {std_dev:.2f}')
    plt.legend()
    plt.show()

    print(f'Average number of bodies per book: {avg}')
    print(f'Median number of bodies per book: {median}')
    print(f'Standard deviation of bodies per book: {std_dev}')

# Show number of texts per book, with average/median/standard deviation
def show_texts_per_book(annotations):
    texts_per_book = []

    for annotation in annotations.values():
        num_texts = sum([len(page.get('text', [])) for page in annotation['page']])
        texts_per_book.append(num_texts)

    # Calculate statistics
    texts_per_book = np.array(texts_per_book)
    avg = np.mean(texts_per_book)
    median = np.median(texts_per_book)
    std_dev = np.std(texts_per_book)

    # Plot bar graph descending order and show average, median, and standard deviation on the graph as horizontal dotted lines
    sorted_indices = np.argsort(texts_per_book)[::-1]
    sorted_texts_per_book = texts_per_book[sorted_indices]
    book_titles = list(annotations.keys())
    sorted_book_titles = [book_titles[i] for i in sorted_indices]

    plt.figure(figsize=(10, 5))
    plt.bar(sorted_book_titles, sorted_texts_per_book)
    plt.xticks(rotation=90)
    plt.title('Number of Texts per Book')
    plt.xlabel('Book')
    plt.ylabel('Number of Texts')
    plt.grid(axis='y')
    plt.axhline(y=avg, color='r', linestyle='--', linewidth=2, label=f'Average: {avg:.2f}')
    plt.axhline(y=median, color='g', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
    plt.axhline(y=std_dev, color='b', linestyle='--', linewidth=2, label=f'Std Dev: {std_dev:.2f}')
    plt.legend()
    plt.show()

    print(f'Average number of texts per book: {avg}')
    print(f'Median number of texts per book: {median}')
    print(f'Standard deviation of texts per book: {std_dev}')

# Show number of annotations per book, with average/median/standard deviation as stacked bar graph for frames, faces, bodies, texts
def show_annotations_per_book(annotations):
    frames_per_book = []
    faces_per_book = []
    bodies_per_book = []
    texts_per_book = []

    for annotation in annotations.values():
        num_frames = sum([len(page.get('frame', [])) for page in annotation['page']])
        num_faces = sum([len(page.get('face', [])) for page in annotation['page']])
        num_bodies = sum([len(page.get('body', [])) for page in annotation['page']])
        num_texts = sum([len(page.get('text', [])) for page in annotation['page']])

        frames_per_book.append(num_frames)
        faces_per_book.append(num_faces)
        bodies_per_book.append(num_bodies)
        texts_per_book.append(num_texts)

    # Calculate statistics
    frames_per_book = np.array(frames_per_book)
    faces_per_book = np.array(faces_per_book)
    bodies_per_book = np.array(bodies_per_book)
    texts_per_book = np.array(texts_per_book)

    # Plot stacked bar graph
    sorted_indices = np.argsort(frames_per_book)[::-1]
    sorted_frames_per_book = frames_per_book[sorted_indices]
    sorted_faces_per_book = faces_per_book[sorted_indices]
    sorted_bodies_per_book = bodies_per_book[sorted_indices]
    sorted_texts_per_book = texts_per_book[sorted_indices]
    book_titles = list(annotations.keys())
    sorted_book_titles = [book_titles[i] for i in sorted_indices]

    plt.figure(figsize=(10, 5))
    plt.bar(sorted_book_titles, sorted_frames_per_book, label='Frames')
    plt.bar(sorted_book_titles, sorted_faces_per_book, bottom=sorted_frames_per_book, label='Faces')
    plt.bar(sorted_book_titles, sorted_bodies_per_book, bottom=sorted_frames_per_book + sorted_faces_per_book, label='Bodies')
    plt.bar(sorted_book_titles, sorted_texts_per_book, bottom=sorted_frames_per_book + sorted_faces_per_book + sorted_bodies_per_book, label='Texts')
    plt.xticks(rotation=90)
    plt.title('Number of Annotations per Book')
    plt.xlabel('Book')
    plt.ylabel('Number of Annotations')
    plt.grid(axis='y')
    plt.legend()
    plt.show()

# TODO: show shapes of all bounding boxes for frames, faces, bodies, texts centered at (0, 0) with width and height normalized to 1 using thickness of lines to show frequency of sizes and create separate graphs for each type of annotation (frame, face, body, text) with different colors for each book and show the average size of bounding boxes for each type of annotation as a thick black line on the graph
def show_bounding_box_shapes(annotations):
    for annotation in annotations.values():
        for page in annotation['page']:
            for frame in page.get('frame', []):
                xmin = int(frame.get('xmin'), [])
                ymin = int(frame.get('ymin'), [])
                xmax = int(frame.get('xmax'), [])
                ymax = int(frame.get('ymax'), [])

                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin

                plt.plot([x_center - width / 2, x_center + width / 2, x_center + width / 2, x_center - width / 2, x_center - width / 2],
                         [y_center - height / 2, y_center - height / 2, y_center + height / 2, y_center + height / 2, y_center - height / 2])

            for face in page.get('face', []):
                xmin = int(face.get('xmin'))
                ymin = int(face.get('ymin'))
                xmax = int(face.get('xmax'))
                ymax = int(face.get('ymax'))

                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin

                plt.plot([x_center - width / 2, x_center + width / 2, x_center + width / 2, x_center - width / 2, x_center - width / 2],
                         [y_center - height / 2, y_center - height / 2, y_center + height / 2, y_center + height / 2, y_center - height / 2])

            for body in page.get('body', []):
                xmin = int(body.get('xmin'))
                ymin = int(body.get('ymin'))
                xmax = int(body.get('xmax'))
                ymax = int(body.get('ymax'))

                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin

                plt.plot([x_center - width / 2, x_center + width / 2, x_center + width / 2, x_center - width / 2, x_center - width / 2],
                         [y_center - height / 2, y_center - height / 2, y_center + height / 2, y_center + height / 2, y_center - height / 2])

            for text in page.get('text', []):
                xmin = int(text.get('xmin'))
                ymin = int(text.get('ymin'))
                xmax = int(text.get('xmax'))
                ymax = int(text.get('ymax'))

                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin

                plt.plot([x_center - width / 2, x_center + width / 2, x_center + width / 2, x_center - width / 2, x_center - width / 2],
                         [y_center - height / 2, y_center - height / 2, y_center + height / 2, y_center + height / 2, y_center - height / 2])

    plt.show()

# TODO: plot distribution of bounding box width, height, aspect ratio, and area for frames, faces, bodies, texts and get average, median, and standard deviation for each type of annotation
def show_bounding_box_statistics(annotations):
    for annotation in annotations.values():
        for page in annotation['page']:
            width = int(page.get('width'))
            height = int(page.get('height'))

            for frame in page.get('frame', []):
                xmin = int(frame.get('xmin'))
                ymin = int(frame.get('ymin'))
                xmax = int(frame.get('xmax'))
                ymax = int(frame.get('ymax'))

                width = xmax - xmin
                height = ymax - ymin
                aspect_ratio = width / height
                area = width * height

            for face in page.get('face', []):
                xmin = int(face.get('xmin'))
                ymin = int(face.get('ymin'))
                xmax = int(face.get('xmax'))
                ymax = int(face.get('ymax'))

                width = xmax - xmin
                height = ymax - ymin
                aspect_ratio = width / height
                area = width * height

            for body in page.get('body', []):
                xmin = int(body.get('xmin'))
                ymin = int(body.get('ymin'))
                xmax = int(body.get('xmax'))
                ymax = int(body.get('ymax'))

                width = xmax - xmin
                height = ymax - ymin
                aspect_ratio = width / height
                area = width * height

            for text in page.get('text', []):
                xmin = int(text.get('xmin'))
                ymin = int(text.get('ymin'))
                xmax = int(text.get('xmax'))
                ymax = int(text.get('ymax'))

                width = xmax - xmin
                height = ymax - ymin
                aspect_ratio = width / height
                area = width * height

    # Calculate statistics
    width = np.array(width)
    height = np.array(height)
    aspect_ratio = np.array(aspect_ratio)
    area = np.array(area)

    avg_width = np.mean(width)
    median_width = np.median(width)
    std_dev_width = np.std(width)

    avg_height = np.mean(height)
    median_height = np.median(height)
    std_dev_height = np.std(height)

    avg_aspect_ratio = np.mean(aspect_ratio)
    median_aspect_ratio = np.median(aspect_ratio)
    std_dev_aspect_ratio = np.std(aspect_ratio)

    avg_area = np.mean(area)
    median_area = np.median(area)
    std_dev_area = np.std(area)

    # Plot bar graph descending order and show average, median, and standard deviation on the graph as horizontal dotted lines
    sorted_indices = np.argsort(width)[::-1]
    sorted_width = width[sorted_indices]
    sorted_height = height[sorted_indices]
    sorted_aspect_ratio = aspect_ratio[sorted_indices]
    sorted_area = area[sorted_indices]

    plt.figure(figsize=(10, 5))
    plt.bar(sorted_indices, sorted_width, label='Width')
    plt.bar(sorted_indices, sorted_height, label='Height')
    plt.bar(sorted_indices, sorted_aspect_ratio, label='Aspect Ratio')
    plt.bar(sorted_indices, sorted_area, label='Area')
    plt.xticks(rotation=90)
    plt.title('Bounding Box Statistics')
    plt.xlabel('Annotation')
    plt.ylabel('Value')
    plt.grid(axis='y')
    plt.axhline(y=avg_width, color='r', linestyle='--', linewidth=2, label=f'Average Width: {avg_width:.2f}')
    plt.axhline(y=median_width, color='g', linestyle='--', linewidth=2, label=f'Median Width: {median_width:.2f}')
    plt.axhline(y=std_dev_width, color='b', linestyle='--', linewidth=2, label=f'Std Dev Width: {std_dev_width:.2f}')
    plt.axhline(y=avg_height, color='r', linestyle='--', linewidth=2, label=f'Average Height: {avg_height:.2f}')
    plt.axhline(y=median_height, color='g', linestyle='--', linewidth=2, label=f'Median Height: {median_height:.2f}')
    plt.axhline(y=std_dev_height, color='b', linestyle='--', linewidth=2, label=f'Std Dev Height: {std_dev_height:.2f}')
    plt.axhline(y=avg_aspect_ratio, color='r', linestyle='--', linewidth=2, label=f'Average Aspect Ratio: {avg_aspect_ratio:.2f}')
    plt.axhline(y=median_aspect_ratio, color='g', linestyle='--', linewidth=2, label=f'Median Aspect Ratio: {median_aspect_ratio:.2f}')
    plt.axhline(y=std_dev_aspect_ratio, color='b', linestyle='--', linewidth=2, label=f'Std Dev Aspect Ratio: {std_dev_aspect_ratio:.2f}')
    plt.axhline(y=avg_area, color='r', linestyle='--', linewidth=2, label=f'Average Area: {avg_area:.2f}')
    plt.axhline(y=median_area, color='g', linestyle='--', linewidth=2, label=f'Median Area: {median_area:.2f}')
    plt.axhline(y=std_dev_area, color='b', linestyle='--', linewidth=2, label=f'Std Dev Area: {std_dev_area:.2f}')
    plt.legend()
    plt.show()

    print(f'Average width: {avg_width}')
    print(f'Median width: {median_width}')
    print(f'Standard deviation of width: {std_dev_width}')
    print(f'Average height: {avg_height}')
    print(f'Median height: {median_height}')
    print(f'Standard deviation of height: {std_dev_height}')
    print(f'Average aspect ratio: {avg_aspect_ratio}')
    print(f'Median aspect ratio: {median_aspect_ratio}')
    print(f'Standard deviation of aspect ratio: {std_dev_aspect_ratio}')
    print(f'Average area: {avg_area}')
    print(f'Median area: {median_area}')
    print(f'Standard deviation of area: {std_dev_area}')


# Function to calculate statistics for a given DataFrame
def calculate_statistics(df):
    stats = {
        'images_count': len(df),
        'frames_count': df['type'].value_counts().get('frame', 0),
        'text_count': df['type'].value_counts().get('text', 0),
        'faces_count': df['type'].value_counts().get('face', 0),
        'bodies_count': df['type'].value_counts().get('body', 0),
        'min_width_frames': df['width'].min(),
        'max_width_frames': df['width'].max(),
        'avg_width_frames': df['width'].mean(),
        'median_width_frames': df['width'].median(),
        'std_width_frames': df['width'].std(),
        'min_height_frames': df['height'].min(),
        'max_height_frames': df['height'].max(),
        'avg_height_frames': df['height'].mean(),
        'median_height_frames': df['height'].median(),
        'std_height_frames': df['height'].std(),
        'min_area_frames': df['area'].min(),
        'max_area_frames': df['area'].max(),
        'avg_area_frames': df['area'].mean(),
        'median_area_frames': df['area'].median(),
        'std_area_frames': df['area'].std(),
        'min_aspect_ratio_frames': df['aspect_ratio'].min(),
        'max_aspect_ratio_frames': df['aspect_ratio'].max(),
        'avg_aspect_ratio_frames': df['aspect_ratio'].mean(),
        'median_aspect_ratio_frames': df['aspect_ratio'].median(),
        'std_aspect_ratio_frames': df['aspect_ratio'].std()
    }
    return stats



def main():
    ROOT_DIR = '../Manga109'
    IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
    ANNOTATIONS_DIR = os.path.join(ROOT_DIR, 'annotations')
    ANNOTATIONS_CSV = os.path.join(ROOT_DIR, 'Manga109.csv')


    dataset_statistics(annotations)
    show_images_per_book(annotations)
    show_characters_per_book(annotations)
    show_frames_per_book(annotations)
    show_faces_per_book(annotations)
    show_bodies_per_book(annotations)
    show_texts_per_book(annotations)
    show_annotations_per_book(annotations)
    show_bounding_box_shapes(annotations)



if __name__ == "__main__":
    main()
