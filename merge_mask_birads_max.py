
import os
import cv2
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm

def merge_masks_with_intensity(df, input_folder, output_folder, index_split):
    os.makedirs(output_folder, exist_ok=True)

    # Split DataFrame into two parts
    if index_split > 0 and index_split < len(df):
        df_part1 = df.iloc[:index_split]
        df_part2 = df.iloc[index_split:]
    else:
        print("Invalid index_split value. Using the entire DataFrame.")
        df_part1 = df
        df_part2 = pd.DataFrame()

    # Process part 1
    if not df_part1.empty:
        merge_masks_part(df_part1, input_folder, output_folder, "Part 1")

    # Process part 2
    if not df_part2.empty:
        merge_masks_part(df_part2, input_folder, output_folder, "Part 2")

def merge_masks_part(df_part, input_folder, output_folder, desc):
    # Dictionary to store masks for each ID
    id_masks = {}

    for index, row in tqdm(df_part.iterrows(), desc=f"Processing masks - {desc}", total=len(df_part), unit="mask"):
        image_id = row['image_id']
        mask_id = row['mask_id']
        birads = row['birads']

        # Construct the original mask path
        original_mask_path = os.path.join(input_folder, mask_id)

        try:
            # Load the original mask
            original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)  # Intensity values [0, 255]
            # Normalize intensity values to range [0, 1]
            original_mask = original_mask / 255.0

            # Normalize intensity values to range [0, 1] based on lesion_types
            if birads == 2:
                intensity = 0.25
            elif birads == 3:
                intensity = 0.50
            elif birads == 4:
                intensity = 0.75
            elif birads == 5:
                intensity = 1.0

            original_mask = original_mask * intensity

            # Merge masks for the same ID
            id_number = image_id
            if id_number in id_masks:
                id_masks[id_number] = np.maximum(id_masks[id_number], original_mask)
            else:
                id_masks[id_number] = original_mask

        except Exception as e:
            print(f"Error processing mask {mask_id}: {str(e)}")

    # Save merged masks for each ID
    for id_number, merged_mask in tqdm(id_masks.items(), desc=f"Saving merged masks - {desc}", unit="mask"):
        output_path = os.path.join(output_folder, f"{id_number}")
        # Cap values at 1
        merged_mask[merged_mask > 1] = 1
        cv2.imwrite(output_path, (merged_mask * 255).astype(int))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge masks based on conditions.")
    parser.add_argument("--input_folder", required=True, help="Path to the input folder containing original masks.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for merged masks.")
    parser.add_argument("--csv_path", required=True, help="Path to the CSV file containing DataFrame information.")
    parser.add_argument("--index_split", type=int, default=0, help="Index to split the DataFrame into two parts.")

    args = parser.parse_args()

    # Load the DataFrame
    df = pd.read_csv(args.csv_path)

    # Merge masks with intensity based on lesion_types
    merge_masks_with_intensity(df, args.input_folder, args.output_folder, args.index_split)
