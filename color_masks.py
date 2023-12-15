import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

def colorize_masks(mask_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Check all .png files in the specified mask folder
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]

    # List to store all unique colors
    all_colors = []

    for mask_file in tqdm(mask_files, desc="Collecting unique colors", unit="mask"):
        mask_path = os.path.join(mask_folder, mask_file)

        # Read mask image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Get unique pixel values in the mask
        unique_values = np.unique(mask)

        # Add unique colors to the list
        all_colors.extend(list(unique_values))

    # Remove background color (0)
    all_colors = [color for color in all_colors if color != 0]

    # Create a color palette for visualization
    color_palette = plt.cm.get_cmap('tab10', len(set(all_colors)))

    for mask_file in tqdm(mask_files, desc="Colorizing masks", unit="mask"):
        mask_path = os.path.join(mask_folder, mask_file)

        # Read mask image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Create an RGB image with the same size as the mask
        colorized_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        # Apply different colors to each unique color in the mask
        for i, unique_color in enumerate(set(all_colors)):
            if unique_color != 0:  # Ignore the background color
                color = np.array(color_palette(i)[:3]) * 255
                colorized_mask[mask == unique_color] = color

        # Save the colorized mask to the output folder
        output_path = os.path.join(output_folder, f"colorized_{mask_file}")
        cv2.imwrite(output_path, colorized_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Colorize masks in specified mask folder.')
    parser.add_argument('--mask_folder', help='Path to the mask folder', required=True)
    parser.add_argument('--output_folder', help='Path to the output folder for colorized masks', default='colorized_output')

    args = parser.parse_args()

    colorize_masks(args.mask_folder, args.output_folder)
