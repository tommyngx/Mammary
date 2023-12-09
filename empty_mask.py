
# Update 30 Nov 2023
# Tommy bugs
# Miscellaneous utilities.
#

import os
import argparse
from tqdm import tqdm
import pandas as pd
from PIL import Image

def is_empty_mask(mask_path):
    # Function to check if a mask is empty (all pixel values are 0)
    with Image.open(mask_path) as img:
        array = np.array(img)
        return not array.any()

def process_folder(folder_path):
    # Function to process a folder and create a DataFrame
    data = {'ID': [], 'width': [], 'height': [], 'folder_name': [], 'path': [], 'empty': []}
    
    files = os.listdir(folder_path)
    total_files = len(files)
    
    for file in tqdm(files, desc=f"Processing {folder_path}"):
        file_path = os.path.join(folder_path, file)
        with Image.open(file_path) as img:
            width, height = img.size
        
        data['ID'].append(file)
        data['width'].append(width)
        data['height'].append(height)
        data['folder_name'].append(os.path.basename(folder_path))
        data['path'].append(file_path)
        data['empty'].append(int(is_empty_mask(file_path)))

    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description='Process mask folders and create a DataFrame.')
    parser.add_argument('--folder1', help='Path to the first mask folder', required=True)
    parser.add_argument('--folder2', help='Path to the second mask folder', required=True)
    parser.add_argument('--folder3', help='Path to the third mask folder', required=True)
    parser.add_argument('--export_csv', help='Path to export the CSV file', default='empty_mask.csv')

    args = parser.parse_args()

    folder1_df = process_folder(args.folder1)
    folder2_df = process_folder(args.folder2)
    folder3_df = process_folder(args.folder3)

    result_df = pd.concat([folder1_df, folder2_df, folder3_df], ignore_index=True)

    result_df.to_csv(args.export_csv, index=False)
    print("CSV file exported successfully.")

if __name__ == "__main__":
    main()



