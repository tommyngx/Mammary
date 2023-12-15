import os
from tqdm import tqdm
import shutil

def merge_crop_mask_folders(input_folders, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    images_output_folder = os.path.join(output_folder, "images")
    masks_output_folder = os.path.join(output_folder, "masks")

    os.makedirs(images_output_folder, exist_ok=True)
    os.makedirs(masks_output_folder, exist_ok=True)

    for i, input_folder in enumerate(input_folders, start=1):
        images_folder = os.path.join(input_folder, "images")
        masks_folder = os.path.join(input_folder, "masks")

        # Copy images to the merged images folder
        for image_file in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image_file)
            new_image_path = os.path.join(images_output_folder, image_file)
            shutil.copy(image_path, new_image_path)

        # Copy masks to the merged masks folder with suffix (_1, _2, etc.)
        for mask_file in os.listdir(masks_folder):
            mask_name, mask_extension = os.path.splitext(mask_file)
            new_mask_name = f"{mask_name}_{i}{mask_extension}"
            mask_path = os.path.join(masks_folder, mask_file)
            new_mask_path = os.path.join(masks_output_folder, new_mask_name)
            shutil.copy(mask_path, new_mask_path)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Merge data from crop_mask folders.')
    parser.add_argument('--input_folders', nargs='+', help='Paths to crop_mask folders', required=True)
    parser.add_argument('--output_folder', help='Path to the output folder', required=True)

    args = parser.parse_args()

    merge_crop_mask_folders(args.input_folders, args.output_folder)

if __name__ == "__main__":
    main()
