import os
import cv2
import argparse
from tqdm import tqdm

def merge_masks(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to store masks for each ID
    id_masks = {}

    # Iterate through the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in tqdm(files, desc="Processing masks", unit="mask"):
            if file.lower().endswith('.png'):
                # Extract ID from the file name
                id_number = file.rsplit('_', 1)[0]  # Extract all characters before the last underscore

                # Load the mask
                mask_path = os.path.join(root, file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # Merge masks for the same ID
                if id_number in id_masks:
                    id_masks[id_number] += mask
                else:
                    id_masks[id_number] = mask

    # Save merged masks
    for id_number, merged_mask in id_masks.items():
        output_path = os.path.join(output_folder, f"{id_number}.png")
        cv2.imwrite(output_path, merged_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge masks based on ID.")
    parser.add_argument("--input_folder", required=True, help="Path to the input folder containing masks.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for merged masks.")

    args = parser.parse_args()

    # Merge masks
    merge_masks(args.input_folder, args.output_folder)
