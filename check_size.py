
# Update 30 Nov 2023
# Tommy bugs
# Miscellaneous utilities.
#

import os
from tqdm import tqdm

def check_size_match(images_folder, mask_folder):
    image_files = os.listdir(images_folder)
    mask_files = os.listdir(mask_folder)

    total_files = min(len(image_files), len(mask_files))
    
    for file_name in tqdm(image_files[:total_files], desc="Checking Size Match"):
        image_path = os.path.join(images_folder, file_name)
        mask_path = os.path.join(mask_folder, file_name)

        try:
            image_size = os.path.getsize(image_path)
            mask_size = os.path.getsize(mask_path)

            if image_size != mask_size:
                print(f"Size mismatch for file: {file_name} - Image {image_size} - Mask {mask_size}")
        except FileNotFoundError:
            print(f"File not found in both folders: {file_name}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Check if sizes of corresponding images and masks match.')
    parser.add_argument('images_folder', help='Path to the images folder')
    parser.add_argument('mask_folder', help='Path to the mask folder')

    args = parser.parse_args()

    check_size_match(args.images_folder, args.mask_folder)



