import os
import shutil
import argparse
from tqdm import tqdm

def copy_folder_and_csv(input_folder, csv_file, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Copy folder X to Y
    input_folder_path = os.path.abspath(input_folder)
    output_folder_path = os.path.join(output_folder, "")
    
    # Count the number of files and directories to copy
    total_files = sum([len(files) for _, _, files in os.walk(input_folder_path)]) + 1  # +1 for the CSV file
    
    # Use tqdm to create a progress bar
    with tqdm(total=total_files, desc="Copying files", unit="file") as pbar:
        shutil.copytree(input_folder_path, output_folder_path, copy_function=shutil.copy2, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.DS_Store', '.git'))
        pbar.update(1)  # Update progress bar for the copied folder

        # Copy CSV file to Y
        csv_file_path = os.path.join(output_folder, "", os.path.basename(csv_file))
        shutil.copy(csv_file, csv_file_path)
        pbar.update(1)  # Update progress bar for the copied CSV file

def main():
    parser = argparse.ArgumentParser(description='Copy folder X and a CSV file, then paste them with new names.')
    parser.add_argument('--input_folder', help='Path to the input folder (X)', required=True)
    parser.add_argument('--csv_file', help='Path to the CSV file', required=True)
    parser.add_argument('--output_folder', help='Path to the output folder', default='output')

    args = parser.parse_args()

    copy_folder_and_csv(args.input_folder, args.csv_file, args.output_folder)
    print("Copy completed.")

if __name__ == "__main__":
    main()
