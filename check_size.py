
# Update 30 Nov 2023
# Tommy bugs
# Miscellaneous utilities.
#

import os
from tqdm import tqdm
from PIL import Image

def check_size_match(image_folder, mask_folder):
    image_files = os.listdir(image_folder)
    mask_files = os.listdir(mask_folder)
    
    # Ensure the same files exist in both folders
    common_files = set(image_files) & set(mask_files)

    mismatched_files = []

    for file in tqdm(common_files, desc="Checking file sizes"):
        image_path = os.path.join(image_folder, file)
        mask_path = os.path.join(mask_folder, file)

        with Image.open(image_path) as img:
            image_width, image_height = img.size

        with Image.open(mask_path) as img:
            mask_width, mask_height = img.size

        if image_width != mask_width or image_height != mask_height:
            mismatched_files.append({
                'file': file,
                'image_width': image_width,
                'image_height': image_height,
                'mask_width': mask_width,
                'mask_height': mask_height,
            })

    return mismatched_files

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Check size match between images and masks.')
    parser.add_argument('--images_folder', help='Path to the images folder', required=True)
    parser.add_argument('--mask_folder', help='Path to the masks folder', required=True)

    args = parser.parse_args()

    mismatched_files = check_size_match(args.images_folder, args.mask_folder)

    if not mismatched_files:
        print("All files have matching sizes.")
    else:
        print("Mismatched file details:")
        for mismatch in mismatched_files:
            print(f"File: {mismatch['file']}, Image Size: ({mismatch['image_width']}, {mismatch['image_height']}), "
                  f"Mask Size: ({mismatch['mask_width']}, {mismatch['mask_height']})")

if __name__ == "__main__":
    main()




