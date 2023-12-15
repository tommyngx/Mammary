import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

def colorize_and_blend(mask_path, images_folder, output_folder):
    mask_file = os.path.basename(mask_path)
    image_path = os.path.join(images_folder, mask_file)

    # Read the mask and image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create an RGB image with the same size as the mask
    colorized_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    # Blend the colorized mask with the original image
    blended_image = cv2.addWeighted(image, 0.5, colorized_mask, 0.5, 0)

    # Save the blended image to the output folder
    output_path = os.path.join(output_folder, f"blended_{mask_file}")
    cv2.imwrite(output_path, blended_image)

def colorize_and_blend_all(images_folder, masks_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Get all mask files in the specified folder
    mask_files = [f for f in os.listdir(masks_folder) if f.endswith('.png')]

    # Use tqdm for progress visualization
    for mask_file in tqdm(mask_files, desc="Colorizing and Blending", unit="mask"):
        mask_path = os.path.join(masks_folder, mask_file)
        colorize_and_blend(mask_path, images_folder, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Colorize masks and blend with images.')
    parser.add_argument('--images_folder', help='Path to the images folder', required=True)
    parser.add_argument('--masks_folder', help='Path to the masks folder', required=True)
    parser.add_argument('--output_folder', help='Path to the output folder for blended images', default='blended_output')

    args = parser.parse_args()

    colorize_and_blend_all(args.images_folder, args.masks_folder, args.output_folder)
