
# Update 30 Nov 2023
# Tommy bugs
# Miscellaneous utilities.
#

import os
from tqdm import tqdm
from PIL import Image

def is_empty_mask(mask_path):
    # Function to check if a mask is empty (all pixel values are 0)
    with Image.open(mask_path) as img:
        array = np.array(img)
        return not array.any()

def delete_empty_masks(images_folder, masks_folder):
    image_files = os.listdir(images_folder)
    mask_files = os.listdir(masks_folder)

    # Ensure the same files exist in both folders
    common_files = set(image_files) & set(mask_files)

    deleted_count = 0

    for file in tqdm(common_files, desc="Deleting empty masks"):
        image_path = os.path.join(images_folder, file)
        mask_path = os.path.join(masks_folder, file)

        if os.path.exists(mask_path) and is_empty_mask(mask_path):
            os.remove(mask_path)
            os.remove(image_path)  # Delete corresponding image

            deleted_count += 1

    return deleted_count

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Delete empty masks and corresponding images.')
    parser.add_argument('--images_folder', help='Path to the images folder', required=True)
    parser.add_argument('--masks_folder', help='Path to the masks folder', required=True)

    args = parser.parse_args()

    deleted_count = delete_empty_masks(args.images_folder, args.masks_folder)

    if deleted_count == 0:
        print("No empty masks found and deleted.")
    else:
        print(f"{deleted_count} empty masks and corresponding images deleted.")

if __name__ == "__main__":
    main()
