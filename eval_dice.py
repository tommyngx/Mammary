import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def dice_score(pred, true):
    intersection = np.sum(pred * true)
    return (2. * intersection) / (np.sum(pred) + np.sum(true))

def jaccard_index(pred, true):
    intersection = np.sum(pred * true)
    union = np.sum(pred) + np.sum(true) - intersection
    return intersection / union

def calculate_metrics(folder1, folder2):
    files1 = [f for f in os.listdir(folder1) if f.endswith('.png') or f.endswith('.jpg')]
    files2 = [f for f in os.listdir(folder2) if f.endswith('.png') or f.endswith('.jpg')]

    dice_scores = []
    jaccard_indices = []

    for file1 in tqdm(files1, desc="Calculating metrics"):
        if file1 in files2:
            path1 = os.path.join(folder1, file1)
            path2 = os.path.join(folder2, file1)

            mask1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

            if mask1 is None or mask2 is None:
                print(f"Skipping {file1} due to read error.")
                continue

            # Binarize the masks
            mask1 = (mask1 > 0).astype(np.uint8)
            mask2 = (mask2 > 0).astype(np.uint8)

            dice = dice_score(mask1, mask2)
            jaccard = jaccard_index(mask1, mask2)

            dice_scores.append(dice)
            jaccard_indices.append(jaccard)

    return dice_scores, jaccard_indices

def print_metrics(dice_scores, jaccard_indices):
    if len(dice_scores) == 0 or len(jaccard_indices) == 0:
        print("No metrics to display.")
        return

    print(f"Dice Score: Mean = {np.mean(dice_scores):.4f}, Std = {np.std(dice_scores):.4f}")
    print(f"Jaccard Index: Mean = {np.mean(jaccard_indices):.4f}, Std = {np.std(jaccard_indices):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Dice score and Jaccard index for segmentation masks in two folders")
    parser.add_argument('--folder1', type=str, required=True, help="Path to the first folder containing masks")
    parser.add_argument('--folder2', type=str, required=True, help="Path to the second folder containing masks")
    args = parser.parse_args()

    dice_scores, jaccard_indices = calculate_metrics(args.folder1, args.folder2)
    print_metrics(dice_scores, jaccard_indices)