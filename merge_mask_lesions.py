import os
import cv2
import pandas as pd
import argparse
from tqdm import tqdm

def merge_masks_with_conditions(df, input_folder, output_folder, split_df):
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to store the first mask for each ID
    id_first_masks = {}

    # Only process a subset of the DataFrame based on the split_df value
    start_index = 0 if split_df == 0 else len(df) // 2
    end_index = len(df) // 2 if split_df == 0 else len(df)

    for index, row in tqdm(df.iloc[start_index:end_index].iterrows(), desc="Processing masks", total=(end_index - start_index), unit="mask"):
        image_id = row['image_id']
        mask_id = row['mask_id']
        lesion_type = row['lesion_types']

        # Construct the original mask path
        original_mask_path = os.path.join(input_folder, mask_id)

        try:
            # Load the original mask
            original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)  # Intensity values [0, 255]

            # Normalize intensity values to range [0, 1]
            original_mask = original_mask / 255.0

            # Merge masks based on conditions
            if lesion_type == 'Mass':
                intensity = 0.75
            elif lesion_type == 'Architecturaldistorsion':
                intensity = 0.25
            elif lesion_type == 'Asymmetry':
                intensity = 0.50
            elif lesion_type == 'Microcalcification':
                intensity = 1.0

            # Add the first mask for each ID to the dictionary
            if image_id not in id_first_masks:
                id_first_masks[image_id] = original_mask * intensity

        except Exception as e:
            print(f"Error processing mask {mask_id}: {str(e)}")

    # Save merged masks for each ID
    for image_id, first_mask in tqdm(id_first_masks.items(), desc="Saving merged masks", unit="mask"):
        output_path = os.path.join(output_folder, f"{image_id}")
        cv2.imwrite(output_path, (first_mask * 255).astype(int))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge masks based on conditions.")
    parser.add_argument("--input_folder", required=True, help="Path to the input folder containing original masks.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for merged masks.")
    parser.add_argument("--csv_path", required=True, help="Path to the CSV file containing DataFrame information.")
    parser.add_argument("--split_df", type=int, default=0, choices=[0, 1], help="Split the DataFrame into two parts. Use 0 for the first half, and 1 for the second half.")

    args = parser.parse_args()

    # Load the DataFrame
    df = pd.read_csv(args.csv_path)

    # Merge masks based on conditions
    merge_masks_with_conditions(df, args.input_folder, args.output_folder, args.split_df)
