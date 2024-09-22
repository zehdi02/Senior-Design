import os
import json
import logging
import manga109api
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import defaultdict


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROCESSED_BOOKS_FILE = "processed_books.json"

def load_processed_books():
    """Load the list of processed books from file."""
    if os.path.exists(PROCESSED_BOOKS_FILE):
        with open(PROCESSED_BOOKS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_processed_books(processed_books):
    """Save the list of processed books to file."""
    with open(PROCESSED_BOOKS_FILE, 'w') as f:
        json.dump(processed_books, f, indent=4)

def convert_numpy_to_serializable(obj):
    """Helper function to convert NumPy objects to JSON serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):  # Convert NumPy integer
        return int(obj)
    if isinstance(obj, np.floating):  # Convert NumPy float
        return float(obj)
    return obj


def load_annotations(parser, books):
    logging.info("Loading annotations for books...")
    annotations = {}
    for book in books:
        logging.info(f"Loading annotations for book: {book}")
        annotations[book] = parser.get_annotation(book=book)
    logging.info("Finished loading annotations.")
    return annotations


def calculate_statistics(annotations, processed_books):
    logging.info("Calculating statistics...")
    stats = defaultdict(lambda: defaultdict(list))
    occurrence_stats = defaultdict(lambda: defaultdict(list))
    density_stats = defaultdict(list)
    inter_object_distances = defaultdict(lambda: defaultdict(list))
    frame_content_analysis = defaultdict(lambda: defaultdict(list))

    for book, annotation in annotations.items():
        # Skip books that have already been processed
        if book in processed_books:
            logging.info(f"Skipping already processed book: {book}")
            continue

        logging.info(f"Processing book: {book}")
        for page_idx, page in enumerate(annotation['page']):
            width = int(page['@width'])
            height = int(page['@height'])

            occurrence_per_page = {'body': 0, 'face': 0, 'frame': 0, 'text': 0}
            objects_per_frame = []

            for annotation_type in ['body', 'face', 'frame', 'text']:
                if annotation_type in page:
                    for roi in page[annotation_type]:
                        xmin = int(roi['@xmin'])
                        ymin = int(roi['@ymin'])
                        xmax = int(roi['@xmax'])
                        ymax = int(roi['@ymax'])

                        bbox_width = xmax - xmin
                        bbox_height = ymax - ymin
                        aspect_ratio = bbox_width / bbox_height

                        # Collect statistics on spatial features
                        stats['all']['min_x'].append(xmin)
                        stats['all']['min_y'].append(ymin)
                        stats['all']['width'].append(bbox_width)
                        stats['all']['height'].append(bbox_height)
                        stats['all']['aspect_ratio'].append(aspect_ratio)

                        stats[book]['min_x'].append(xmin)
                        stats[book]['min_y'].append(ymin)
                        stats[book]['width'].append(bbox_width)
                        stats[book]['height'].append(bbox_height)
                        stats[book]['aspect_ratio'].append(aspect_ratio)

                        occurrence_per_page[annotation_type] += 1
                        objects_per_frame.append((xmin, ymin, annotation_type))

                        # Bounding Box Size Distribution
                        stats[annotation_type]['bounding_box_size'].append(bbox_width * bbox_height)

            page_area = width * height
            density_stats[book].append({
                'body_density': occurrence_per_page['body'] / page_area,
                'face_density': occurrence_per_page['face'] / page_area,
                'frame_density': occurrence_per_page['frame'] / page_area,
                'text_density': occurrence_per_page['text'] / page_area,
            })

            # Calculate inter-object distances
            for i in range(len(objects_per_frame)):
                for j in range(i + 1, len(objects_per_frame)):
                    obj1, obj2 = objects_per_frame[i], objects_per_frame[j]
                    dist = distance.euclidean((obj1[0], obj1[1]), (obj2[0], obj2[1]))
                    inter_object_distances[book][f"{obj1[2]}-{obj2[2]}"].append(dist)

            frame_content_analysis[book][page_idx] = occurrence_per_page.copy()
            occurrence_stats['page'][book].append(occurrence_per_page)
            occurrence_stats['book'][book].append(occurrence_per_page)

        # After processing the book, mark it as processed
        processed_books.append(book)

    logging.info("Finished calculating statistics.")
    return stats, occurrence_stats, density_stats, inter_object_distances, frame_content_analysis


def calculate_summary_statistics(stats):
    summary = {'all': {}}

    # Convert NumPy arrays to lists for JSON serialization
    def convert_to_list(arr):
        return arr.tolist() if isinstance(arr, np.ndarray) else arr

    # Calculate for all books combined
    for key, values in stats['all'].items():
        summary['all'][key] = {
            'mean': convert_to_list(np.mean(values)),
            'median': convert_to_list(np.median(values)),
            'std': convert_to_list(np.std(values)),
            'distribution': [convert_to_list(x) for x in np.histogram(values, bins=50)]  # Convert histogram
        }

    # Calculate by individual book
    for book, book_stats in stats.items():
        if book == 'all':  # Skip aggregate statistics for individual books
            continue
        summary[book] = {}
        for key, values in book_stats.items():
            summary[book][key] = {
                'mean': convert_to_list(np.mean(values)),
                'median': convert_to_list(np.median(values)),
                'std': convert_to_list(np.std(values)),
                'distribution': [convert_to_list(x) for x in np.histogram(values, bins=50)]  # Convert histogram
            }

    return summary


def calculate_occurrence_statistics(occurrence_stats):
    logging.info("Calculating occurrence statistics...")
    occurrence_summary = {'page': {}, 'book': {}}

    for book, occurrences in occurrence_stats['page'].items():
        total_occurrences = {'body': 0, 'face': 0, 'frame': 0, 'text': 0}
        for page_stats in occurrences:
            for key, value in page_stats.items():
                total_occurrences[key] += value

        occurrence_summary['page'][book] = {
            'mean': {key: np.mean([p[key] for p in occurrences]) for key in total_occurrences},
            'median': {key: np.median([p[key] for p in occurrences]) for key in total_occurrences},
            'std': {key: np.std([p[key] for p in occurrences]) for key in total_occurrences}
        }

    logging.info("Finished calculating occurrence statistics.")
    return occurrence_summary


def calculate_density_statistics(density_stats):
    logging.info("Calculating density statistics...")
    density_summary = {}

    for book, densities in density_stats.items():
        density_summary[book] = {
            'mean_body_density': np.mean([d['body_density'] for d in densities]),
            'mean_face_density': np.mean([d['face_density'] for d in densities]),
            'mean_frame_density': np.mean([d['frame_density'] for d in densities]),
            'mean_text_density': np.mean([d['text_density'] for d in densities])
        }

    logging.info("Finished calculating density statistics.")
    return density_summary


def calculate_inter_object_distance(inter_object_distances):
    logging.info("Calculating inter-object distances...")
    distance_summary = {}

    for book, distances in inter_object_distances.items():
        distance_summary[book] = {}
        for pair, dist_list in distances.items():
            distance_summary[book][pair] = {
                'mean': np.mean(dist_list),
                'median': np.median(dist_list),
                'std': np.std(dist_list)
            }

    logging.info("Finished calculating inter-object distances.")
    return distance_summary


def calculate_frame_content_analysis(frame_content_analysis):
    logging.info("Calculating frame content analysis...")
    content_summary = {}

    for book, frame_data in frame_content_analysis.items():
        content_summary[book] = {}
        for frame, occurrences in frame_data.items():
            total_objects = sum(occurrences.values())
            content_summary[book][frame] = {
                'total_objects': total_objects,
                'object_breakdown': occurrences
            }

    logging.info("Finished calculating frame content analysis.")
    return content_summary


def analyze_object_area_distribution(output_folder):
    with open(f"{output_folder}/summary_statistics.json", "r") as f:
        summary_stats = json.load(f)

    # Extract bounding box sizes from the summary statistics for each type
    bbox_sizes = {
        "text": summary_stats["text"]["bounding_box_size"]["distribution"][0],
        "body": summary_stats["body"]["bounding_box_size"]["distribution"][0],
        "face": summary_stats["face"]["bounding_box_size"]["distribution"][0],
        "frame": summary_stats["frame"]["bounding_box_size"]["distribution"][0]
    }

    # Plot the distribution of bounding box sizes for each type
    plt.figure(figsize=(10, 6))
    for obj_type, sizes in bbox_sizes.items():
        plt.hist(sizes, bins=50, alpha=0.5, label=f"{obj_type} area")

    plt.title('Object Area Distribution')
    plt.xlabel('Bounding Box Area (width * height)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def analyze_page_density(output_folder):
    with open(f"{output_folder}/density_statistics.json", "r") as f:
        density_stats = json.load(f)

    # Extract density values for different object types
    body_densities = []
    face_densities = []
    frame_densities = []
    text_densities = []

    for book, density_list in density_stats.items():
        logging.info(f"Processing density statistics for book: {book}")
        # Check that each entry in the density list is a dictionary
        for d in density_list:
            if isinstance(d, dict):  # Make sure d is a dictionary
                body_densities.append(d.get('mean_body_density', 0))
                face_densities.append(d.get('mean_face_density', 0))
                frame_densities.append(d.get('mean_frame_density', 0))
                text_densities.append(d.get('mean_text_density', 0))
            else:
                logging.warning(f"Unexpected entry in density list for book {book}: {d}")


    # Plot densities
    plt.figure(figsize=(10, 6))
    plt.hist(body_densities, bins=50, alpha=0.5, label="Body Density")
    plt.hist(face_densities, bins=50, alpha=0.5, label="Face Density")
    plt.hist(frame_densities, bins=50, alpha=0.5, label="Frame Density")
    plt.hist(text_densities, bins=50, alpha=0.5, label="Text Density")

    plt.title('Page Object Density Distribution')
    plt.xlabel('Object Density (objects per unit area)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def analyze_object_inter_relationships(output_folder):
    with open(f"{output_folder}/distance_statistics.json", "r") as f:
        distance_stats = json.load(f)

    # Store all distances between different object pairs
    distances = {
        "text-body": [],
        "text-face": [],
        "text-frame": [],
        "body-face": [],
        "body-frame": [],
        "face-frame": []
    }

    for book, distance_data in distance_stats.items():
        for pair, dist_list in distance_data.items():
            if pair in distances:
                distances[pair].extend(dist_list['mean'])  # Collect all mean distances

    # Plot the distributions of distances
    plt.figure(figsize=(10, 6))
    for pair, dist_values in distances.items():
        plt.hist(dist_values, bins=50, alpha=0.5, label=f"{pair} distance")

    plt.title('Inter-Object Distance Distribution')
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def analyze_aspect_ratio(output_folder):
    with open(f"{output_folder}/summary_statistics.json", "r") as f:
        summary_stats = json.load(f)

    # Extract aspect ratios from the summary statistics for each type
    aspect_ratios = {
        "text": summary_stats["text"]["aspect_ratio"]["distribution"][0],
        "body": summary_stats["body"]["aspect_ratio"]["distribution"][0],
        "face": summary_stats["face"]["aspect_ratio"]["distribution"][0],
        "frame": summary_stats["frame"]["aspect_ratio"]["distribution"][0]
    }

    # Plot the distribution of aspect ratios for each type
    plt.figure(figsize=(10, 6))
    for obj_type, ratios in aspect_ratios.items():
        plt.hist(ratios, bins=50, alpha=0.5, label=f"{obj_type} aspect ratio")

    plt.title('Aspect Ratio Distribution')
    plt.xlabel('Aspect Ratio (width / height)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def main():
    logging.info("Starting main process...")
    manga109_root_dir = "../Manga109"
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    parser = manga109api.Parser(root_dir=manga109_root_dir)
    books = parser.books

    # Load the list of already processed books
    processed_books = load_processed_books()

    # Load annotations and calculate statistics
    annotations = load_annotations(parser, books)
    stats, occurrence_stats, density_stats, inter_object_distances, frame_content_analysis = calculate_statistics(annotations, processed_books)

    summary = calculate_summary_statistics(stats)
    occurrence_summary = calculate_occurrence_statistics(occurrence_stats)
    density_summary = calculate_density_statistics(density_stats)
    distance_summary = calculate_inter_object_distance(inter_object_distances)
    content_summary = calculate_frame_content_analysis(frame_content_analysis)

    with open(os.path.join(output_folder, 'summary_statistics.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    with open(os.path.join(output_folder, 'occurrence_statistics.json'), 'w') as f:
        json.dump(occurrence_summary, f, indent=4)

    with open(os.path.join(output_folder, 'density_statistics.json'), 'w') as f:
        json.dump(density_summary, f, indent=4)

    with open(os.path.join(output_folder, 'distance_statistics.json'), 'w') as f:
        json.dump(distance_summary, f, indent=4)

    with open(os.path.join(output_folder, 'frame_content_analysis.json'), 'w') as f:
        json.dump(content_summary, f, indent=4)

    save_processed_books(processed_books)
    logging.info("Statistics and graphs have been saved to the output folder.")
    logging.info("Main process finished.")

    analyze_object_area_distribution(output_folder)
    analyze_page_density(output_folder)
    analyze_object_inter_relationships(output_folder)
    analyze_aspect_ratio(output_folder)




if __name__ == "__main__":
    main()