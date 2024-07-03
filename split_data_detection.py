import os
import shutil
import random
import argparse
from tqdm import tqdm
import cv2

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

def split_dataset(image_folder, label_folder, output_folder, train_ratio=0.8):
    # Ensure the output directories exist
    train_image_dir = os.path.join(output_folder, 'train', 'images')
    train_label_dir = os.path.join(output_folder, 'train', 'labels')
    valid_image_dir = os.path.join(output_folder, 'valid', 'images')
    valid_label_dir = os.path.join(output_folder, 'valid', 'labels')
    
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(valid_image_dir, exist_ok=True)
    os.makedirs(valid_label_dir, exist_ok=True)

    # List all images and labels
    images = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
    labels = [f for f in os.listdir(label_folder) if f.endswith('.xml') or f.endswith('.txt')]

    # Ensure each image has a corresponding label
    images.sort()
    labels.sort()
    paired_files = list(zip(images, labels))

    # Shuffle the data
    random.shuffle(paired_files)

    # Split data into training and validation sets
    split_index = int(len(paired_files) * train_ratio)
    train_files = paired_files[:split_index]
    valid_files = paired_files[split_index:]

    def copy_and_resize(files, image_dir, label_dir):
        for image_file, label_file in tqdm(files, desc=f"Copying and resizing files to {image_dir}"):
            image_path = os.path.join(image_folder, image_file)
            label_path = os.path.join(label_folder, label_file)

            # Read image and label
            image = cv2.imread(image_path)
            with open(label_path, 'r') as f:
                label_content = f.readlines()

            # Parse label content
            bboxes = []
            for line in label_content:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    x_min = (x_center - width / 2) * image.shape[1]
                    y_min = (y_center - height / 2) * image.shape[0]
                    x_max = (x_center + width / 2) * image.shape[1]
                    y_max = (y_center + height / 2) * image.shape[0]
                    bboxes.append((x_min, y_min, x_max, y_max))

            # Resize image and adjust bounding boxes
            resized_image, resized_bboxes = resize_and_adjust_bboxes(image, bboxes)

            # Save resized image
            resized_image_path = os.path.join(image_dir, image_file)
            cv2.imwrite(resized_image_path, resized_image)

            # Save adjusted label
            resized_label_path = os.path.join(label_dir, label_file)
            with open(resized_label_path, 'w') as f:
                for bbox in resized_bboxes:
                    x_min, y_min, x_max, y_max = bbox
                    x_center = (x_min + x_max) / 2 / 640
                    y_center = (y_min + y_max) / 2 / 640
                    width = (x_max - x_min) / 640
                    height = (y_max - y_min) / 640
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")

    # Process and copy training files
    copy_and_resize(train_files, train_image_dir, train_label_dir)

    # Process and copy validation files
    copy_and_resize(valid_files, valid_image_dir, valid_label_dir)

    print(f"Dataset split completed. Training set: {len(train_files)} samples, Validation set: {len(valid_files)} samples.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the folder containing image files")
    parser.add_argument('--label_folder', type=str, required=True, help="Path to the folder containing label files")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save the split dataset")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="Ratio of the training set size to the total dataset size")
    args = parser.parse_args()

    split_dataset(args.image_folder, args.label_folder, args.output_folder, args.train_ratio)
