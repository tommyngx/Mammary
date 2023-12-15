import os
import argparse
import numpy as np
from tqdm import tqdm

def analyze_masks(mask_folder):
    # Get list of mask folders
    mask_folders = [f for f in os.listdir(mask_folder) if os.path.isdir(os.path.join(mask_folder, f))]

    # Initialize variables
    all_unique_pixels = set()
    coverage_percentages = []

    # Use tqdm for progress tracking
    for mask_subfolder in tqdm(mask_folders, desc="Analyzing masks", unit="mask"):
        mask_subfolder_path = os.path.join(mask_folder, mask_subfolder)
        mask_files = [f for f in os.listdir(mask_subfolder_path) if f.endswith('.png')]

        for mask_file in mask_files:
            mask_path = os.path.join(mask_subfolder_path, mask_file)
            unique_pixels, coverage_percentage = analyze_mask(mask_path)

            # Update unique pixels
            all_unique_pixels.update(unique_pixels)

            # Update coverage percentages
            coverage_percentages.append(coverage_percentage)

    # Convert set to array for easier analysis
    unique_pixels_array = np.array(list(all_unique_pixels))

    # Calculate statistics
    max_coverage = np.max(coverage_percentages)
    min_coverage = np.min(coverage_percentages)
    avg_coverage = np.mean(coverage_percentages)

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
