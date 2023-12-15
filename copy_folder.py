import os
import shutil
import argparse

def copy_folder_and_csv(input_folder, csv_file, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Copy folder X to Y
    input_folder_path = os.path.abspath(input_folder)
    output_folder_path = os.path.join(output_folder, "Y")
    shutil.copytree(input_folder_path, output_folder_path, copy_function=shutil.copy2)

    # Copy CSV file to Y
    csv_file_path = os.path.join(output_folder, "Y", os.path.basename(csv_file))
    shutil.copy(csv_file, csv_file_path)

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
