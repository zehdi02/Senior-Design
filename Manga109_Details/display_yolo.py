import cv2

# Function to display image with bounding boxes
def display_image_with_bboxes_yolo(full_image_path, full_label_path):
    # Read the image
    img = cv2.imread(full_image_path)

    # Check if image is loaded successfully
    if img is None:
        print(f"Error loading image: {full_image_path}")
        return

    # Read labels and draw bounding boxes
    with open(full_label_path, 'r') as label_file:
        for line in label_file.readlines():
            data = line.strip().split()
            class_id = int(data[0])
            x_center = float(data[1]) * img.shape[1]
            y_center = float(data[2]) * img.shape[0]
            bbox_width = float(data[3]) * img.shape[1]
            bbox_height = float(data[4]) * img.shape[0]

            # Calculate the top-left and bottom-right corners
            x_min = int(x_center - bbox_width / 2)
            y_min = int(y_center - bbox_height / 2)
            x_max = int(x_center + bbox_width / 2)
            y_max = int(y_center + bbox_height / 2)

            # Draw the bounding box on the image
            color = (0, 255, 0)  # Green box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(img, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the image
    cv2.imshow("Image with Bounding Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = "../bounder_YOLO/Manga109_YOLO/train/images/AisazuNihaIrarenai_003_left.jpg"
    label_path = "../bounder_YOLO/Manga109_YOLO/train/labels/AisazuNihaIrarenai_003_left.txt"
    display_image_with_bboxes_yolo(image_path, label_path)

if __name__ == "__main__":
    main()