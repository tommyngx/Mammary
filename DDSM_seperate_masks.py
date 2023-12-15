import os
from tqdm import tqdm
import shutil

def separate_masks(images_folder, masks_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    mask_files = os.listdir(masks_folder)

    for mask_file in tqdm(mask_files, desc="Separating masks"):
        mask_path = os.path.join(masks_folder, mask_file)

        # Extract mask name without extension
        mask_name_parts = os.path.splitext(mask_file)[0].rsplit('_', 1)
        mask_name = mask_name_parts[0]

        # Extract suffix (_1, _2, _3, etc.)
        suffix = mask_name_parts[-1]

        # Create output folder for the specific suffix
        output_suffix_folder = os.path.join(output_folder, f"mask_{suffix}")
        os.makedirs(output_suffix_folder, exist_ok=True)

        # Create subfolders for images and masks
        output_images_folder = os.path.join(output_suffix_folder, "images")
        output_masks_folder = os.path.join(output_suffix_folder, "masks")
        os.makedirs(output_images_folder, exist_ok=True)
        os.makedirs(output_masks_folder, exist_ok=True)

        # Copy mask to the corresponding masks folder
        new_mask_name = f"{mask_name}.png"
        new_mask_path = os.path.join(output_masks_folder, new_mask_name)
        shutil.copy(mask_path, new_mask_path)

        # Copy corresponding image to the images folder
        for image_file in os.listdir(images_folder):
            image_name = os.path.splitext(image_file)[0]
            if image_name.startswith(mask_name):
                image_path = os.path.join(images_folder, image_file)
                new_image_path = os.path.join(output_images_folder, f"{image_name}.png")
                shutil.copy(image_path, new_image_path)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Separate masks into folders based on suffix (_1, _2, _3, etc.).')
    parser.add_argument('--images_folder', help='Path to the images folder', required=True)
    parser.add_argument('--masks_folder', help='Path to the masks folder', required=True)
    parser.add_argument('--output_folder', help='Path to the output folder', required=True)

    args = parser.parse_args()

    separate_masks(args.images_folder, args.masks_folder, args.output_folder)

if __name__ == "__main__":
    main()
