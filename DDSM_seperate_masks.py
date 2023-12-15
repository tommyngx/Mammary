import os
import shutil

def separate_masks(input_folder, output_folder_prefix):
    mask_files = os.listdir(input_folder)

    for mask_file in mask_files:
        # Extract mask name without extension
        mask_name, ext = os.path.splitext(mask_file)

        # Split the mask name by '_'
        mask_name_parts = mask_name.split('_')

        # Extract the suffix (e.g., '1', '2', '3', '4') from the last part of the name
        suffix = mask_name_parts[-1]

        # Create output folder based on the suffix
        output_folder = f"{output_folder_prefix}_{suffix}"

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Copy the mask to the appropriate output folder
        output_mask_path = os.path.join(output_folder, f"{mask_name}{ext}")
        shutil.copy(os.path.join(input_folder, mask_file), output_mask_path)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Separate masks into different folders based on suffix.')
    parser.add_argument('--input_folder', help='Path to the input masks folder', required=True)
    parser.add_argument('--output_folder_prefix', help='Prefix for the output folders', required=True)

    args = parser.parse_args()

    separate_masks(args.input_folder, args.output_folder_prefix)

if __name__ == "__main__":
    main()
