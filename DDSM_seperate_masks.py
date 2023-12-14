
# Update 30 Nov 2023
# Tommy bugs
# Miscellaneous utilities.
#
import os
from tqdm import tqdm
from shutil import copyfile

def separate_masks(mask_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    mask_files = os.listdir(mask_folder)

    for mask_file in tqdm(mask_files, desc="Separating masks"):
        mask_path = os.path.join(mask_folder, mask_file)

        # Split the mask name by "_"
        mask_name_parts = mask_file.split("_")

        if len(mask_name_parts) > 1:
            # Extract the last part after splitting by "_"
            suffix = mask_name_parts[-1]

            # Remove any numeric suffix like "_1", "_2"
            suffix = "".join(filter(str.isalpha, suffix))

            # Create a folder for each suffix and copy the mask
            output_subfolder = os.path.join(output_folder, suffix)
            os.makedirs(output_subfolder, exist_ok=True)

            # Construct the new mask name without numeric suffix
            new_mask_name = "_".join(mask_name_parts[:-1]) + ".png"
            new_mask_path = os.path.join(output_subfolder, new_mask_name)

            # Copy the mask to the corresponding folder
            copyfile(mask_path, new_mask_path)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Separate masks into folders based on suffixes.')
    parser.add_argument('--mask_folder', help='Path to the mask folder', required=True)
    parser.add_argument('--output_folder', help='Path to the output folder', required=True)

    args = parser.parse_args()

    separate_masks(args.mask_folder, args.output_folder)

if __name__ == "__main__":
    main()
