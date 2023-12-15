import os
import shutil
import argparse
from tqdm import tqdm

def copy_folder_and_csv(images_folder, masks_folder, csv_file, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Copy images to output folder
    copy_files(images_folder, os.path.join(output_folder, "images"))

    # Copy masks to output folder
    copy_files(masks_folder, os.path.join(output_folder, "masks"))

    # Copy CSV file to output folder
    shutil.copy(csv_file, output_folder)

def copy_files(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of files to copy
    files_to_copy = os.listdir(input_folder)

    # Use tqdm for progress tracking
    for file in tqdm(files_to_copy, desc=f"Copying files to {output_folder}", unit="file"):
        source_path = os.path.join(input_folder, file)
        destination_path = os.path.join(output_folder, file)
        shutil.copy(source_path, destination_path)

def main():
    parser = argparse.ArgumentParser(description='Copy images, masks, and a CSV file to an output folder.')
    parser.add_argument('--images_folder', help='Path to the images folder', required=True)
    parser.add_argument('--masks_folder', help='Path to the masks folder', required=True)
    parser.add_argument('--csv_file', help='Path to the CSV file', required=True)
    parser.add_argument('--output_folder', help='Path to the output folder', default='output')

    args = parser.parse_args()

    copy_folder_and_csv(args.images_folder, args.masks_folder, args.csv_file, args.output_folder)
    print("Copy completed.")

if __name__ == "__main__":
    main()
