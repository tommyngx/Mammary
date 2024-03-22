import os
import cv2
import argparse
from tqdm import tqdm
import shutil

import os
import cv2
import argparse
from tqdm import tqdm

def resize_images(input_folder, save_folder, size):
    """Function to resize images while keeping the aspect ratio"""

    # Create save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Iterate over the files in the input folder
    for image_name in tqdm(os.listdir(input_folder), desc="Resizing images"):
        image_path = os.path.join(input_folder, image_name)
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        # Get original image dimensions
        height, width = image.shape[:2]

        # Calculate aspect ratio
        aspect_ratio = width / height

        # Calculate new dimensions based on specified width
        new_width = size
        new_height = int(new_width / aspect_ratio)

        # Resize image while keeping aspect ratio
        resized_image = cv2.resize(image, (new_width, new_height))

        # Save resized image
        save_path = os.path.join(save_folder, image_name)
        cv2.imwrite(save_path, resized_image)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize images while keeping aspect ratio')
    parser.add_argument('--input_folder', required=True, help='Input folder containing images')
    parser.add_argument('--save_folder', required=True, help='Folder to save resized images')
    parser.add_argument('--size', type=int, default=640, help='Width of the resized images')
    args = parser.parse_args()

    resize_images(args.input_folder, args.save_folder, args.size)

