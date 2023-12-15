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

    for mask_file in tqdm(mask_files, desc="Colorizing masks", unit="mask"):
        mask_path = os.path.join(mask_folder, mask_file)

        # Read mask image
        mask = np.array(Image.open(mask_path))

        # Get unique pixel values in the mask excluding background (0)
        unique_values = np.unique(mask)[1:]

        # Create a color palette for visualization
        color_palette = plt.cm.get_cmap('tab10', len(unique_values))

        # Create a color map based on the unique values
        cmap = ListedColormap([color_palette(i) for i in range(len(unique_values))])

        # Visualize the mask with color
        plt.imshow(mask, cmap=cmap, vmin=unique_values.min(), vmax=unique_values.max())
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
