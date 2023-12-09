import os
import cv2
import pandas as pd
import argparse
from tqdm import tqdm

def merge_masks_with_conditions(df, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to store masks for each ID
    id_masks = {}

    # Apply intensity to all images based on the lesion_types column
    for index, row in tqdm(df.iterrows(), desc="Applying intensity to images", total=len(df), unit="image"):
        image_id = row['image_id']
        mask_id = row['mask_id']
        lesion_type = row['lesion_types']

        # Construct the original mask path
        original_mask_path = os.path.join(input_folder, mask_id)

        try:
            # Load the original mask
            original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)  # Intensity values [0, 255]

            # Normalize intensity values to range [0, 1] based on lesion type
            if lesion_type == 'Mass':
                intensity = 0.25
            elif lesion_type == 'Architecturaldistorsion':
                intensity = 0.75
            elif lesion_type == 'Asymmetry':
                intensity = 0.50
            elif lesion_type == 'Microcalcification':
                intensity = 1.0

            # Apply intensity to the image
            original_mask = original_mask * intensity

            # Merge masks for the same ID
            id_number = mask_id.rsplit('_', 1)[0]
            if id_number in id_masks:
                id_masks[id_number] += original_mask
            else:
                id_masks[id_number] = original_mask

        except Exception as e:
            print(f"Error processing mask {mask_id}: {str(e)}")

    # Save merged masks
    for id_number, merged_mask in tqdm(id_masks.items(), desc="Saving merged masks", unit="mask"):
        # Set values higher than 1 to 1
        merged_mask[merged_mask > 1] = 1

        output_path = os.path.join(output_folder, f"{id_number}.png")
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
