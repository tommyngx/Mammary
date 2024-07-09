import os
import cv2
import argparse
from tqdm import tqdm
import numpy as np

def adjust_bboxes(bboxes, image_shape):
    height, width = image_shape[:2]
    adjusted_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # Increase size by 30%
        x_min = max(0, x_min - bbox_width * 0.3)
        y_min = max(0, y_min - bbox_height * 0.3)
        x_max = min(width, x_max + bbox_width * 0.3)
        y_max = min(height, y_max + bbox_height * 0.3)

        adjusted_bboxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
    return adjusted_bboxes

def find_bboxes_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(contour) for contour in contours]
    return [(x, y, x + w, y + h) for x, y, w, h in bboxes]

def resize_and_paste_pred_mask(original_mask, pred_mask, bbox):
    x_min, y_min, x_max, y_max = bbox
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    resized_pred_mask = cv2.resize(pred_mask, (bbox_width, bbox_height))
    
    # Ensure the resized_pred_mask has the same number of channels as the original_mask
    if len(original_mask.shape) == 3 and len(resized_pred_mask.shape) == 2:
        resized_pred_mask = cv2.cvtColor(resized_pred_mask, cv2.COLOR_GRAY2BGR)
    
    new_mask = original_mask.copy()
    new_mask[y_min:y_max, x_min:x_max] = resized_pred_mask

    return new_mask

def crop_and_save(image, mask, pred_mask_folder, bboxes, output_image_dir, output_mask_dir, predict_ori_mask_dir, base_filename, resize_to=None):
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        if resize_to:
            cropped_image = cv2.resize(cropped_image, resize_to)
            cropped_mask = cv2.resize(cropped_mask, resize_to)

        pred_mask_path = os.path.join(pred_mask_folder, f"{base_filename}.png")
        if os.path.exists(pred_mask_path):
            pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
            new_mask = resize_and_paste_pred_mask(mask, pred_mask, bbox)
            predict_ori_mask_path = os.path.join(predict_ori_mask_dir, f"{base_filename}_predictOriMask_{i}.png")
            cv2.imwrite(predict_ori_mask_path, new_mask)

        image_filename = os.path.join(output_image_dir, f"{base_filename}_crop_{i}.png")
        mask_filename = os.path.join(output_mask_dir, f"{base_filename}_crop_{i}.png")
        cv2.imwrite(image_filename, cropped_image)
        cv2.imwrite(mask_filename, cropped_mask)

def process_images_and_masks(image_folder, mask_folder, pred_mask_folder, output_folder, resize_to):
    output_image_dir = os.path.join(output_folder, 'images')
    output_mask_dir = os.path.join(output_folder, 'masks')
    predict_ori_mask_dir = os.path.join(output_folder, 'predictOriMask')

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(predict_ori_mask_dir, exist_ok=True)

    images = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
    masks = [f for f in os.listdir(mask_folder) if f.endswith('.png') or f.endswith('.jpg')]

    for image_file in tqdm(images, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, image_file)

        if not os.path.exists(mask_path):
            continue

        # Read image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Find bounding boxes from the mask
        bboxes = find_bboxes_from_mask(mask)

        # Adjust bounding boxes
        adjusted_bboxes = adjust_bboxes(bboxes, image.shape)

        # Crop and save images and masks
        base_filename = os.path.splitext(image_file)[0]
        crop_and_save(image, mask, pred_mask_folder, adjusted_bboxes, output_image_dir, output_mask_dir, predict_ori_mask_dir, base_filename, resize_to)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop regions from images and masks based on bounding boxes")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the folder containing image files")
    parser.add_argument('--mask_folder', type=str, required=True, help="Path to the folder containing mask files")
    parser.add_argument('--pred_mask_folder', type=str, required=True, help="Path to the folder containing predicted masks")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save cropped images and masks")
    parser.add_argument('--resize_to', type=int, nargs=2, default=None, help="Resize cropped images and masks to this size (width height)")
    args = parser.parse_args()

    process_images_and_masks(args.image_folder, args.mask_folder, args.pred_mask_folder, args.output_folder, tuple(args.resize_to) if args.resize_to else None)
