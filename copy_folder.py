import os
import shutil
import argparse

def copy_and_rename_folder_and_csv(input_folder, csv_file, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Copy the entire folder to the output folder
    shutil.copytree(input_folder, os.path.join(output_folder, 'new_folder'))

    # Copy the CSV file to the output folder
    shutil.copy(csv_file, os.path.join(output_folder, 'new_csv.csv'))

def main():
    parser = argparse.ArgumentParser(description='Copy folder and CSV file to a new location with a new name.')
    parser.add_argument('--input_folder', help='Path to the input folder', required=True)
    parser.add_argument('--csv_file', help='Path to the CSV file', required=True)
    parser.add_argument('--output_folder', help='Path to the output folder', required=True)

    args = parser.parse_args()

    copy_and_rename_folder_and_csv(args.input_folder, args.csv_file, args.output_folder)

if __name__ == "__main__":
    main()
