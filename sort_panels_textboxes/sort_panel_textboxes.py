import os
import numpy as np
import networkx as nx
from shapely.geometry import box
from copy import deepcopy
from itertools import groupby

def load_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            annotations.append((class_id, (x_min, y_min, x_max, y_max)))
    return annotations

def erode_rectangle(rect, factor=0.5):
    x_min, y_min, x_max, y_max = rect
    dx = (x_max - x_min) * factor
    dy = (y_max - y_min) * factor
    return (x_min + dx, y_min + dy, x_max - dx, y_max - dy)

# sort panels using topological sorting
def sort_panels(rects):
    rects = [erode_rectangle(rect, 0.05) for rect in rects]
    G = nx.DiGraph()
    G.add_nodes_from(range(len(rects)))

    for i in range(len(rects)):
        for j in range(len(rects)):
            if i == j:
                continue
            if i != j and is_there_a_directed_edge(i, j, rects):
                G.add_edge(i, j, weight=get_distance(rects[i], rects[j]))
            else:
                G.add_edge(j, i, weight=get_distance(rects[i], rects[j]))

    while True:
        cycles = sorted(nx.simple_cycles(G))
        cycles = [cycle for cycle in cycles if len(cycle) > 1]
        if len(cycles) == 0:
            break
        cycle = cycles[0]
        edges = [(cycle[k], cycle[(k + 1) % len(cycle)]) for k in range(len(cycle))]
        max_edge = max(edges, key=lambda x: G.edges[x]["weight"])
        G.remove_edge(*max_edge)

    return list(nx.topological_sort(G))

def is_strictly_above(rectA, rectB):
    x1A, y1A, x2A, y2A = rectA
    x1B, y1B, x2B, y2B = rectB
    return y2A < y1B

def is_strictly_below(rectA, rectB):
    x1A, y1A, x2A, y2A = rectA
    x1B, y1B, x2B, y2B = rectB
    return y2B < y1A

def is_strictly_left_of(rectA, rectB):
    x1A, y1A, x2A, y2A = rectA
    x1B, y1B, x2B, y2B = rectB
    return x2A < x1B

def is_strictly_right_of(rectA, rectB):
    x1A, y1A, x2A, y2A = rectA
    x1B, y1B, x2B, y2B = rectB
    return x2B < x1A

def intersects(rectA, rectB):
    return box(*rectA).intersects(box(*rectB))

def is_there_a_directed_edge(a, b, rects):
    rectA = rects[a]
    rectB = rects[b]
    centre_of_A = [rectA[0] + (rectA[2] - rectA[0]) / 2, rectA[1] + (rectA[3] - rectA[1]) / 2]
    centre_of_B = [rectB[0] + (rectB[2] - rectB[0]) / 2, rectB[1] + (rectB[3] - rectB[1]) / 2]
    if np.allclose(np.array(centre_of_A), np.array(centre_of_B)):
        return box(*rectA).area > (box(*rectB)).area
    copy_A = [rectA[0], rectA[1], rectA[2], rectA[3]]
    copy_B = [rectB[0], rectB[1], rectB[2], rectB[3]]
    while True:
        if is_strictly_above(copy_A, copy_B) and not is_strictly_left_of(copy_A, copy_B):
            return 1
        if is_strictly_above(copy_B, copy_A) and not is_strictly_left_of(copy_B, copy_A):
            return 0
        if is_strictly_right_of(copy_A, copy_B) and not is_strictly_below(copy_A, copy_B):
            return 1
        if is_strictly_right_of(copy_B, copy_A) and not is_strictly_below(copy_B, copy_A):
            return 0
        if is_strictly_below(copy_A, copy_B) and is_strictly_right_of(copy_A, copy_B):
            return use_cuts_to_determine_edge_from_a_to_b(a, b, rects)
        if is_strictly_below(copy_B, copy_A) and is_strictly_right_of(copy_B, copy_A):
           return use_cuts_to_determine_edge_from_a_to_b(a, b, rects)
        # otherwise they intersect
        copy_A = erode_rectangle(copy_A, 0.05)
        copy_B = erode_rectangle(copy_B, 0.05)

def use_cuts_to_determine_edge_from_a_to_b(a, b, rects):
    rects = deepcopy(rects)
    while True:
        xmin, ymin, xmax, ymax = min(rects[a][0], rects[b][0]), min(rects[a][1], rects[b][1]), max(rects[a][2], rects[b][2]), max(rects[a][3], rects[b][3])
        rect_index = [i for i in range(len(rects)) if intersects(rects[i], [xmin, ymin, xmax, ymax])]
        rects_copy = [rect for rect in rects if intersects(rect, [xmin, ymin, xmax, ymax])]
        
        # try to split the panels using a "horizontal" lines
        overlapping_y_ranges = merge_overlapping_ranges([(y1, y2) for x1, y1, x2, y2 in rects_copy])
        panel_index_to_split = {}
        for split_index, (y1, y2) in enumerate(overlapping_y_ranges):
            for i, index in enumerate(rect_index):
                if y1 <= rects_copy[i][1] <= rects_copy[i][3] <= y2:
                    panel_index_to_split[index] = split_index
        
        if panel_index_to_split[a] != panel_index_to_split[b]:
            return panel_index_to_split[a] < panel_index_to_split[b]
        
        # try to split the panels using a "vertical" lines
        overlapping_x_ranges = merge_overlapping_ranges([(x1, x2) for x1, y1, x2, y2 in rects_copy])
        panel_index_to_split = {}
        for split_index, (x1, x2) in enumerate(overlapping_x_ranges[::-1]):
            for i, index in enumerate(rect_index):
                if x1 <= rects_copy[i][0] <= rects_copy[i][2] <= x2:
                    panel_index_to_split[index] = split_index
        if panel_index_to_split[a] != panel_index_to_split[b]:
            return panel_index_to_split[a] < panel_index_to_split[b]
        
        # otherwise, erode the rectangles and try again
        rects = [erode_rectangle(rect, 0.05) for rect in rects]

def merge_overlapping_ranges(ranges):
    """
    ranges: list of tuples (x1, x2)
    """
    if len(ranges) == 0:
        return []
    ranges = sorted(ranges, key=lambda x: x[0])
    merged_ranges = []
    for i, r in enumerate(ranges):
        if i == 0:
            prev_x1, prev_x2 = r
            continue
        x1, x2 = r
        if x1 > prev_x2:
            merged_ranges.append((prev_x1, prev_x2))
            prev_x1, prev_x2 = x1, x2
        else:
            prev_x2 = max(prev_x2, x2)
    merged_ranges.append((prev_x1, prev_x2))
    return merged_ranges

# get distance between two rectangles
def get_distance(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return abs(x1 - x3) + abs(y1 - y3)

def sort_text_boxes_in_reading_order(text_bboxes, sorted_panel_bboxes):
    sorted_text_indices = []
    for panel_index, panel_bbox in enumerate(sorted_panel_bboxes):
        # get the top right corner of the current panel
        panel_x_max = panel_bbox[2]
        panel_y_min = panel_bbox[1]

        # filter text boxes within current panel
        panel_text_indices = [
            i for i, text_bbox in enumerate(text_bboxes)
            if box(*text_bbox).intersects(box(*panel_bbox))
        ]
        
        panel_text_boxes = [text_bboxes[i] for i in panel_text_indices]

        # sort text boxes based on their distance to the panel's top right corner
        sorted_indices = sorted(
            panel_text_indices,
            key=lambda i: get_distance_to_top_right(text_bboxes[i], panel_x_max, panel_y_min)
        )
        
        sorted_text_indices.extend(sorted_indices)

    return sorted_text_indices

def get_distance_to_top_right(text_bbox, panel_x_max, panel_y_min):
    text_x_center = (text_bbox[0] + text_bbox[2]) / 2
    text_y_center = (text_bbox[1] + text_bbox[3]) / 2
    
    distance = ((text_x_center - panel_x_max) ** 2 + (text_y_center - panel_y_min) ** 2) ** 0.5
    return distance

def get_text_to_panel_mapping(text_bboxes, sorted_panel_bboxes):
    text_to_panel_mapping = []
    for text_bbox in text_bboxes:
        shapely_text_polygon = box(*text_bbox)
        all_intersections = []
        all_distances = []
        for j, panel_bbox in enumerate(sorted_panel_bboxes):
            shapely_panel_polygon = box(*panel_bbox)
            if shapely_text_polygon.intersects(shapely_panel_polygon):
                all_intersections.append((shapely_text_polygon.intersection(shapely_panel_polygon).area, j))
            all_distances.append((shapely_text_polygon.distance(shapely_panel_polygon), j))

        if all_intersections:
            text_to_panel_mapping.append(max(all_intersections, key=lambda x: x[0])[1])
        else:
            text_to_panel_mapping.append(min(all_distances, key=lambda x: x[0])[1])
    return text_to_panel_mapping

def save_sorted_annotations(file_path, sorted_indices, annotations):
    with open(file_path, 'w') as f:
        for index in sorted_indices:
            class_id, bbox = annotations[index]
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def main():
    manga_file_name = 'LancelotFullThrottle_036_left'
    
    detected_panels_folder = "C:/Users/Zed/Desktop/CCNY Classes/2024 FALL/CSC 59867 Senior Project II/Project/Senior-Design/detected_panels/"
    panels_path = os.path.join(detected_panels_folder, f"panels/{manga_file_name}_panels_annotations.txt")
    text_boxes_path = os.path.join(detected_panels_folder, f"text_boxes/{manga_file_name}_textboxes_annotations.txt")

    panel_annotations = [(class_id, bbox) for class_id, bbox in load_annotations(panels_path) if class_id == 3]
    text_annotations = [(class_id, bbox) for class_id, bbox in load_annotations(text_boxes_path) if class_id == 2]

    # extract bounding boxes
    panel_bboxes = [bbox for _, bbox in panel_annotations]
    text_bboxes = [bbox for _, bbox in text_annotations]

    # sort panels
    sorted_panel_indices = sort_panels(panel_bboxes)

    # map text boxes to panels and sort within each panel
    panel_id_for_text = get_text_to_panel_mapping(text_bboxes, [panel_bboxes[i] for i in sorted_panel_indices])
    sorted_text_indices = sort_text_boxes_in_reading_order(text_bboxes, [panel_bboxes[i] for i in sorted_panel_indices])

    save_sorted_annotations(os.path.join(detected_panels_folder, "annotation_sorted_panel.txt"), sorted_panel_indices, panel_annotations)
    save_sorted_annotations(os.path.join(detected_panels_folder, "annotation_sorted_textboxes.txt"), sorted_text_indices, text_annotations)

if __name__ == "__main__":
    main()
