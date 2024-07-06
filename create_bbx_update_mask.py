import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def find_bboxes_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(contour) for contour in contours]
    return [(x, y, x + w, y + h, w, h) for x, y, w, h in bboxes]

def process_masks(mask_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    masks = [f for f in os.listdir(mask_folder) if f.endswith('.png') or f.endswith('.jpg')]

    for mask_file in tqdm(masks, desc="Processing masks"):
        mask_path = os.path.join(mask_folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        bboxes = find_bboxes_from_mask(mask)

        for bbox in bboxes:
            x_min, y_min, x_max, y_max, w, h = bbox

            if w < 50 and h < 50:
                mask[y_min:y_max, x_min:x_max] = 0  # Fill with black

        output_mask_path = os.path.join(output_folder, mask_file)
        cv2.imwrite(output_mask_path, mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process masks to remove small bounding boxes")
    parser.add_argument('--mask_folder', type=str, required=True, help="Path to the folder containing mask files")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save updated masks")
    args = parser.parse_args()

    process_masks(args.mask_folder, args.output_folder)