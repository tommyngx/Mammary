import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def create_color_palette(num_colors):
    colors = np.random.randint(0, 256, size=(num_colors, 3), dtype=np.uint8)
    return colors

def process_mask(mask_path, images_folder, output_folder, color_mapping):
    mask_file = os.path.basename(mask_path)
    image_path = os.path.join(images_folder, mask_file)

    # Read the mask and image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)

    # Create an RGB image with the same size as the mask
    colorized_mask = np.zeros_like(image)

    # Find unique values in the mask
    unique_values = np.unique(mask)
    unique_values = unique_values[unique_values != 0]

    # Replace unique values with colors
    for unique_value in unique_values:
        colorized_mask[mask == unique_value] = color_mapping[unique_value]

    # Blend the colorized mask with the original image
    blended_image = cv2.addWeighted(image, 0.7, colorized_mask, 0.3, 0)

    # Save the blended image to the output folder
    output_path = os.path.join(output_folder, f"blended_{mask_file}")
    cv2.imwrite(output_path, blended_image)

def colorize_masks(images_folder, masks_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Create a color palette with 5 different colors
    color_palette = create_color_palette(5)

    # Create a dictionary to map unique values to colors
    color_mapping = {i + 1: color for i, color in enumerate(color_palette)}

    # Use ThreadPoolExecutor to parallelize the processing of masks
    with ThreadPoolExecutor() as executor:
        futures = []
        mask_files = [f for f in os.listdir(masks_folder) if f.endswith('.png')]
        for mask_file in mask_files:
            mask_path = os.path.join(masks_folder, mask_file)
            future = executor.submit(process_mask, mask_path, images_folder, output_folder, color_mapping)
            futures.append(future)

        # Wait for all threads to finish
        tqdm(futures, desc="Colorizing masks", unit="mask")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Colorize masks and apply to images.')
    parser.add_argument('--images_folder', help='Path to the images folder', required=True)
    parser.add_argument('--masks_folder', help='Path to the masks folder', required=True)
    parser.add_argument('--output_folder', help='Path to the output folder for blended images', default='blended_output')

    args = parser.parse_args()

    colorize_masks(args.images_folder, args.masks_folder, args.output_folder)
