import os
import shutil
import argparse

def copy_folder_and_csv(input_folder, csv_file, output_folder, new_name):
    # Create a new folder with the specified new name
    new_folder_path = os.path.join(output_folder, new_name)
    os.makedirs(new_folder_path, exist_ok=True)

    # Copy the entire folder
    shutil.copytree(input_folder, os.path.join(new_folder_path, os.path.basename(input_folder)))

    # Copy the CSV file to the new folder
    shutil.copy(csv_file, new_folder_path)

def main():
    parser = argparse.ArgumentParser(description='Copy a folder and CSV file, then paste with a new name.')
    parser.add_argument('--input_folder', help='Path to the input folder to copy', required=True)
    parser.add_argument('--csv_file', help='Path to the CSV file to copy', required=True)
    parser.add_argument('--output_folder', help='Path to the output folder for the new copy', required=True)
    parser.add_argument('--new_name', help='New name for the copied folder and CSV file', required=True)

    args = parser.parse_args()

    copy_folder_and_csv(args.input_folder, args.csv_file, args.output_folder, args.new_name)

if __name__ == "__main__":
    main()
