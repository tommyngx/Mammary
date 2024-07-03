import os
import cv2
import numpy as np
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree
from tqdm import tqdm
import argparse

# Function to create VOC annotation XML
def create_voc_annotation(img_name, height, width, bboxes):
    annotation = Element('annotation')
    folder = SubElement(annotation, 'folder')
    folder.text = 'VOC'
    filename = SubElement(annotation, 'filename')
    filename.text = img_name
    size = SubElement(annotation, 'size')
    width_tag = SubElement(size, 'width')
    width_tag.text = str(width)
    height_tag = SubElement(size, 'height')
    height_tag.text = str(height)
    depth_tag = SubElement(size, 'depth')
    depth_tag.text = '3'
    for bbox in bboxes:
        obj = SubElement(annotation, 'object')
        name = SubElement(obj, 'name')
        name.text = 'cancer'
        bndbox = SubElement(obj, 'bndbox')
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = str(bbox[0])
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(bbox[1])
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(bbox[2])
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(bbox[3])
    return annotation

# Function to create YOLO annotation text
def create_yolo_annotation(height, width, bboxes):
    yolo_annots = []
    for bbox in bboxes:
        x_center = (bbox[0] + bbox[2]) / 2 / width
        y_center = (bbox[1] + bbox[3]) / 2 / height
        bbox_width = (bbox[2] - bbox[0]) / width
        bbox_height = (bbox[3] - bbox[1]) / height
        yolo_annots.append(f"0 {x_center} {y_center} {bbox_width} {bbox_height}")
    return '\n'.join(yolo_annots)

def process_masks(mask_folder, output_folder):
    # Initialize an empty list to store the bounding box data
    bbox_data = []

    # Iterate over mask images and compute bounding boxes
    for mask_filename in tqdm(os.listdir(mask_folder)):
        if mask_filename.endswith('.png'):
            mask_path = os.path.join(mask_folder, mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            height, width = mask.shape

            # Process the mask: set pixels < 100 to 0, >= 100 to 255
            _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

            # Find contours and bounding boxes
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = [cv2.boundingRect(c) for c in contours]

            # Filter out small bounding boxes
            min_size = 0.001  # 0.1% of the image size
            bboxes = [bbox for bbox in bboxes if bbox[2] >= width * min_size and bbox[3] >= height * min_size]

            # Add data to the list
            for bbox in bboxes:
                x, y, w, h = bbox
                bbox_data.append([mask_filename, height, width, x, y, w, h])

            # Save VOC annotation
            voc_annotation = create_voc_annotation(mask_filename, height, width, bboxes)
            voc_path = os.path.join(output_folder, 'VOC', f"{os.path.splitext(mask_filename)[0]}.xml")
            os.makedirs(os.path.dirname(voc_path), exist_ok=True)
            with open(voc_path, 'wb') as f:
                ElementTree(voc_annotation).write(f)

            # Save YOLO annotation
            yolo_annotation = create_yolo_annotation(height, width, bboxes)
            yolo_path = os.path.join(output_folder, 'YOLO', f"{os.path.splitext(mask_filename)[0]}.txt")
            os.makedirs(os.path.dirname(yolo_path), exist_ok=True)
            with open(yolo_path, 'w') as f:
                f.write(yolo_annotation)

            # Draw bounding boxes on the original mask image
            annotated_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            for bbox in bboxes:
                x, y, w, h = bbox
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            annotated_image_path = os.path.join(output_folder, 'AnnotatedMasks', mask_filename)
            os.makedirs(os.path.dirname(annotated_image_path), exist_ok=True)
            cv2.imwrite(annotated_image_path, annotated_image)

    # Convert bbox data to a DataFrame and save to CSV
    bbox_df = pd.DataFrame(bbox_data, columns=['name', 'height', 'width', 'x', 'y', 'bbox_width', 'bbox_height'])
    csv_path = os.path.join(output_folder, 'bounding_boxes.csv')
    bbox_df.to_csv(csv_path, index=False)

    print(f"Annotations and bounding boxes saved to {output_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert mask images to bounding boxes and annotations")
    parser.add_argument('--mask_folder', type=str, required=True, help="Path to the folder containing mask images")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save output annotations and CSV")
    args = parser.parse_args()

    process_masks(args.mask_folder, args.output_folder)
