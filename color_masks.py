import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def colorize_masks(mask_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Check all .png files in the specified mask folder
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]

    # Define a color palette for visualization
    color_palette = [
        (0, 0, 0),    # Black for background (pixel value 0)
        (255, 0, 0),  # Red for unique value 1
        (0, 255, 0),  # Green for unique value 2
        (0, 0, 255),  # Blue for unique value 3
        (255, 255, 0),  # Yellow for unique value 4
        (255, 0, 255),  # Magenta for unique value 5
    ]

    for mask_file in tqdm(mask_files, desc="Colorizing masks", unit="mask"):
        mask_path = os.path.join(mask_folder, mask_file)

        # Read mask image
        mask = np.array(Image.open(mask_path))

        # Create a color map based on the defined palette
        cmap = ListedColormap(color_palette)

        # Visualize the mask with color
        plt.imshow(mask, cmap=cmap, vmin=0, vmax=len(color_palette) - 1)
        plt.axis('off')

        # Save the colorized mask to the output folder
        output_path = os.path.join(output_folder, f"colorized_{mask_file}")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Colorize masks in specified mask folder.')
    parser.add_argument('--mask_folder', help='Path to the mask folder', required=True)
    parser.add_argument('--output_folder', help='Path to the output folder for colorized masks', default='colorized_output')

    args = parser.parse_args()

    colorize_masks(args.mask_folder, args.output_folder)
