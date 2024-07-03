import os
import shutil
import random
import argparse
from tqdm import tqdm

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

    # Copy files to the respective directories
    for image_file, label_file in tqdm(train_files, desc="Copying training files"):
        shutil.copy(os.path.join(image_folder, image_file), os.path.join(train_image_dir, image_file))
        shutil.copy(os.path.join(label_folder, label_file), os.path.join(train_label_dir, label_file))

    for image_file, label_file in tqdm(valid_files, desc="Copying validation files"):
        shutil.copy(os.path.join(image_folder, image_file), os.path.join(valid_image_dir, image_file))
        shutil.copy(os.path.join(label_folder, label_file), os.path.join(valid_label_dir, label_file))

    print(f"Dataset split completed. Training set: {len(train_files)} samples, Validation set: {len(valid_files)} samples.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the folder containing image files")
    parser.add_argument('--label_folder', type=str, required=True, help="Path to the folder containing label files")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save the split dataset")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="Ratio of the training set size to the total dataset size")
    args = parser.parse_args()

    split_dataset(args.image_folder, args.label_folder, args.output_folder, args.train_ratio)
