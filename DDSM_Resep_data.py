import os
from tqdm import tqdm
import shutil

def merge_crop_mask_data(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    images_output_folder = os.path.join(output_folder, "images")
    masks_output_folder = os.path.join(output_folder, "masks")
    os.makedirs(images_output_folder, exist_ok=True)
    os.makedirs(masks_output_folder, exist_ok=True)

    crop_mask_folders = [folder_name for folder_name in os.listdir(input_folder) if folder_name.startswith("crop_mask_") and os.path.isdir(os.path.join(input_folder, folder_name))]

    for folder_name in tqdm(crop_mask_folders, desc="Merging data", unit="folder"):
        images_folder = os.path.join(input_folder, folder_name, "images")
        masks_folder = os.path.join(input_folder, folder_name, "masks")

        # Copy images to the output images folder
        for image_file in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image_file)
            new_image_path = os.path.join(images_output_folder, image_file)
            shutil.copy(image_path, new_image_path)

        # Copy masks to the output masks folder with added suffix
        for mask_file in os.listdir(masks_folder):
            mask_suffix = folder_name.rsplit("_", 1)[-1]
            new_mask_name = f"{os.path.splitext(mask_file)[0]}_{mask_suffix}.png"
            mask_path = os.path.join(masks_folder, mask_file)
            new_mask_path = os.path.join(masks_output_folder, new_mask_name)
            shutil.copy(mask_path, new_mask_path)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Merge data from crop_mask_number folders.')
    parser.add_argument('--input_folder', help='Path to the input folder', required=True)
    parser.add_argument('--output_folder', help='Path to the output folder', required=True)

    args = parser.parse_args()

    merge_crop_mask_data(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
