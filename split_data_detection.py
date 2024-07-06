import os
import shutil
import random
import argparse
import pandas as pd
from tqdm import tqdm
import cv2
import xml.etree.ElementTree as ET

def resize_and_adjust_bboxes(image, bboxes, target_size=(640, 640)):
    height, width = image.shape[:2]
    target_width, target_height = target_size

    # Resize image
    resized_image = cv2.resize(image, (target_width, target_height))

    # Calculate scale factors
    x_scale = target_width / width
    y_scale = target_height / height

    # Adjust bounding boxes
    resized_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min = int(x_min * x_scale)
        y_min = int(y_min * y_scale)
        x_max = int(x_max * x_scale)
        y_max = int(y_max * y_scale)

        # Ensure x_min < x_max and y_min < y_max
        if x_max <= x_min:
            x_max = x_min + 1
        if y_max <= y_min:
            y_max = y_min + 1

        resized_bboxes.append((x_min, y_min, x_max, y_max))

    return resized_image, resized_bboxes

def parse_yolo_label_file(label_path, image_shape):
    with open(label_path, 'r') as f:
        label_content = f.readlines()
    bboxes = []
    for line in label_content:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id, x_center, y_center, width, height = map(float, parts)
            x_min = (x_center - width / 2) * image_shape[1]
            y_min = (y_center - height / 2) * image_shape[0]
            x_max = (x_center + width / 2) * image_shape[1]
            y_max = (y_center + height / 2) * image_shape[0]
            bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes

def parse_voc_label_file(label_path):
    tree = ET.parse(label_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        x_min = float(bbox.find('xmin').text)
        y_min = float(bbox.find('ymin').text)
        x_max = float(bbox.find('xmax').text)
        y_max = float(bbox.find('ymax').text)
        bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes

def save_voc_label_file(label_path, image_file, bboxes, image_shape):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'VOC'
    ET.SubElement(annotation, 'filename').text = image_file
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(image_shape[1])
    ET.SubElement(size, 'height').text = str(image_shape[0])
    ET.SubElement(size, 'depth').text = '3'

    for bbox in bboxes:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = 'cancer'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(bbox[0])
        ET.SubElement(bndbox, 'ymin').text = str(bbox[1])
        ET.SubElement(bndbox, 'xmax').text = str(bbox[2])
        ET.SubElement(bndbox, 'ymax').text = str(bbox[3])

    tree = ET.ElementTree(annotation)
    tree.write(label_path)

def draw_bboxes_on_image(image, bboxes):
    annotated_image = image.copy()
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return annotated_image

def copy_and_resize(files, image_dir, label_dir, annotated_dir, csv_data, image_folder, label_folder):
    for image_file, label_file in files:
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_file)

        # Read image and label
        image = cv2.imread(image_path)
        if label_file.endswith('.txt'):
            bboxes = parse_yolo_label_file(label_path, image.shape)
        elif label_file.endswith('.xml'):
            bboxes = parse_voc_label_file(label_path)
        else:
            continue

        # Resize image and adjust bounding boxes
        resized_image, resized_bboxes = resize_and_adjust_bboxes(image, bboxes)

        # Save resized image
        resized_image_path = os.path.join(image_dir, image_file)
        cv2.imwrite(resized_image_path, resized_image)

        # Save adjusted label
        resized_label_path = os.path.join(label_dir, label_file)
        if label_file.endswith('.txt'):
            with open(resized_label_path, 'w') as f:
                for bbox in resized_bboxes:
                    x_min, y_min, x_max, y_max = bbox
                    x_center = (x_min + x_max) / 2 / 640
                    y_center = (y_min + y_max) / 2 / 640
                    width = (x_max - x_min) / 640
                    height = (y_max - y_min) / 640
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")
        elif label_file.endswith('.xml'):
            save_voc_label_file(resized_label_path, image_file, resized_bboxes, resized_image.shape)

        # Draw bounding boxes on image and save annotated image
        annotated_image = draw_bboxes_on_image(resized_image, resized_bboxes)
        annotated_image_path = os.path.join(annotated_dir, image_file)
        cv2.imwrite(annotated_image_path, annotated_image)

        # Collect data for CSV
        for bbox in resized_bboxes:
            x_min, y_min, x_max, y_max = bbox
            csv_data.append([image_file, resized_image.shape[0], resized_image.shape[1], x_min, y_min, x_max - x_min, y_max - y_min])

def split_dataset(image_folder, label_folder, output_folder, train_ratio=0.8):
    # Ensure the output directories exist
    train_image_dir = os.path.join(output_folder, 'train', 'images')
    train_label_dir = os.path.join(output_folder, 'train', 'labels')
    test_image_dir = os.path.join(output_folder, 'test', 'images')
    test_label_dir = os.path.join(output_folder, 'test', 'labels')
    annotated_dir = os.path.join(output_folder, 'annotated_bbx')

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    # List all images and labels
    images = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
    labels = [f for f in os.listdir(label_folder) if f.endswith('.xml') or f.endswith('.txt')]

    # Ensure each image has a corresponding label
    images.sort()
    labels.sort()
    paired_files = list(zip(images, labels))

    # Shuffle the data
    random.shuffle(paired_files)

    # Split data into training and test sets
    split_index = int(len(paired_files) * train_ratio)
    train_files = paired_files[:split_index]
    test_files = paired_files[split_index:]

    # Create a list to store CSV data
    csv_data = []

    # Create a progress bar for the whole process
    with tqdm(total=len(train_files) + len(test_files), desc="Processing dataset") as pbar:
        # Process and copy training files
        copy_and_resize(train_files, train_image_dir, train_label_dir, annotated_dir, csv_data, image_folder, label_folder)
        pbar.update(len(train_files))

        # Process and copy test files
        copy_and_resize(test_files, test_image_dir, test_label_dir, annotated_dir, csv_data, image_folder, label_folder)
        pbar.update(len(test_files))

    # Save CSV file
    csv_df = pd.DataFrame(csv_data, columns=['image', 'height', 'width', 'x', 'y', 'bbox_width', 'bbox_height'])
    csv_path = os.path.join(output_folder, 'annotations.csv')
    csv_df.to_csv(csv_path, index=False)

    print(f"Dataset split completed. Training set: {len(train_files)} samples, Test set: {len(test_files)} samples. Annotations saved to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset into training and test sets")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the folder containing image files")