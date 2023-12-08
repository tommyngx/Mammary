import os
import cv2
import pandas as pd
import argparse
from tqdm import tqdm

def merge_masks_with_conditions(df, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to store masks for each ID
    id_masks = {}

    for index, row in tqdm(df.iterrows(), desc="Processing masks", total=len(df), unit="mask"):
        image_id = row['image_id']
        mask_id = row['mask_id']
        lesion_type = row['lesion_types']
        split = row['split']

        # Construct the original mask path
        original_mask_path = os.path.join(input_folder, mask_id)

        # Check if the file exists
        if not os.path.exists(original_mask_path):
            print(f"File not found: {original_mask_path}")
            continue

        # Load the original mask
        original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE) / 255.0  # Normalize to range [0, 1]

        # Merge masks based on conditions
        if lesion_type == 'Mass':
            merged_mask = original_mask * 0.75
        elif lesion_type == 'Architecturaldistorsion':
            merged_mask = original_mask * 0.25
        elif lesion_type == 'Asymmetry':
            merged_mask = original_mask * 0.50
        elif lesion_type == 'Microcalcification':
            merged_mask = original_mask

        # Add the merged mask to the dictionary for the corresponding ID
        id_key = mask_id.rsplit('_', 1)[0]
        if id_key in id_masks:
            id_masks[id_key] += merged_mask
        else:
            id_masks[id_key] = merged_mask

    # Save merged masks for each ID
    for id_key, merged_mask in id_masks.items():
        output_path = os.path.join(output_folder, f"{id_key}.png")
        cv2.imwrite(output_path, (merged_mask * 255).astype(int))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge masks based on conditions.")
    parser.add_argument("--input_folder", required=True, help="Path to the input folder containing original masks.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for merged masks.")
    parser.add_argument("--csv_path", required=True, help="Path to the CSV file containing DataFrame information.")

    args = parser.parse_args()

    # Load the DataFrame
    df = pd.read_csv(args.csv_path)

    # Merge masks based on conditions
    merge_masks_with_conditions(df, args.input_folder, args.output_folder)
