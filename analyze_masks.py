import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

def analyze_masks(mask_folder):
    # Initialize variables
    all_unique_pixels = set()
    coverage_percentages = []

    # Check all .png files in the specified mask folder
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]

    # Use tqdm for progress tracking
    for mask_file in tqdm(mask_files, desc="Analyzing masks", unit="mask"):
        mask_path = os.path.join(mask_folder, mask_file)
        unique_pixels, coverage_percentage = analyze_mask(mask_path)

        # Update unique pixels
        all_unique_pixels.update(unique_pixels)

        # Update coverage percentages
        coverage_percentages.append(coverage_percentage)

        # Print mask name if coverage is 0
        if coverage_percentage == 0:
            print(f"Mask with coverage 0: {mask_path}")

    # Convert set to array for easier analysis
    unique_pixels_array = np.array(list(all_unique_pixels))

    # Check if the array is not empty
    if coverage_percentages:
        # Calculate statistics
        max_coverage = np.max(coverage_percentages)
        min_coverage = np.min(coverage_percentages)
        avg_coverage = np.mean(coverage_percentages)
    else:
        # Handle case when array is empty
        max_coverage = min_coverage = avg_coverage = 0

    return unique_pixels_array, max_coverage, min_coverage, avg_coverage

def analyze_mask(mask_path):
    # Read mask image
    mask = np.array(Image.open(mask_path))

    # Calculate unique values
    unique_pixels = np.unique(mask)

    # Calculate percentage of coverage
    total_pixels = mask.size
    unique_pixels_count = len(unique_pixels)
    coverage_percentage = (unique_pixels_count / total_pixels) * 100

    return unique_pixels, coverage_percentage

def main():
    parser = argparse.ArgumentParser(description='Analyze masks in specified mask folder.')
    parser.add_argument('--mask_folder', help='Path to the mask folder', required=True)

    args = parser.parse_args()

    unique_pixels, max_coverage, min_coverage, avg_coverage = analyze_masks(args.mask_folder)

    print(f"Unique Pixels: {unique_pixels}")
    print(f"Max Coverage: {max_coverage:.2f}%")
    print(f"Min Coverage: {min_coverage:.2f}%")
    print(f"Avg Coverage: {avg_coverage:.2f}%")

if __name__ == "__main__":
    main()
