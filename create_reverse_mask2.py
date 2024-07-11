import os
import cv2
import argparse
from tqdm import tqdm

def adjust_bboxes(bboxes, image_shape, increase_size):
    height, width = image_shape[:2]
    adjusted_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # Increase size by the specified percentage
        x_min = max(0, x_min - bbox_width * increase_size)
        y_min = max(0, y_min - bbox_height * increase_size)
        x_max = min(width, x_max + bbox_width * increase_size)
        y_max = min(height, y_max + bbox_height * increase_size)

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

def process_pred_masks(pred_mask_folder, mask_folder, predict_ori_mask_dir, ori_mask_dir, increase_size):
    os.makedirs(predict_ori_mask_dir, exist_ok=True)
    os.makedirs(ori_mask_dir, exist_ok=True)

    pred_masks = [f for f in os.listdir(pred_mask_folder) if f.lower().endswith('_pred_prediction.png')]

    for pred_mask_file in tqdm(pred_masks, desc="Processing predicted masks"):
        base_filename = pred_mask_file.replace('_pred_prediction.png', '')
        pred_mask_path = os.path.join(pred_mask_folder, pred_mask_file)
        
        # Find all mask files starting with base_filename
        original_mask_files = [f for f in os.listdir(mask_folder) if f.startswith(base_filename) and f.endswith('.png')]

        for mask_file in original_mask_files:
            mask_path = os.path.join(mask_folder, mask_file)

            if not os.path.exists(mask_path):
                continue

            # Read original mask and predicted mask
            original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

            if original_mask is None or pred_mask is None:
                continue

            # Find bounding boxes from the original mask
            bboxes = find_bboxes_from_mask(original_mask)

            # Adjust bounding boxes
            adjusted_bboxes = adjust_bboxes(bboxes, original_mask.shape, increase_size)

            # Resize and paste predicted masks back to the original mask
            for bbox in adjusted_bboxes:
                new_mask = resize_and_paste_pred_mask(original_mask, pred_mask, bbox)
                predict_ori_mask_path = os.path.join(predict_ori_mask_dir, f"{base_filename}.png")
                cv2.imwrite(predict_ori_mask_path, new_mask)

            # Save the original mask in the oriMask directory
            ori_mask_path = os.path.join(ori_mask_dir, f"{base_filename}.png")
            cv2.imwrite(ori_mask_path, original_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Resize and paste predicted masks back to the original masks")
    parser.add_argument('--pred_mask_folder', type=str, required=True, help="Path to the folder containing predicted masks")
    parser.add_argument('--mask_folder', type=str, required=True, help="Path to the folder containing original masks")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save the new masks with predicted masks pasted")
    parser.add_argument('--increase_size', type=float, default=0.3, help="Percentage to increase the size of the bounding boxes (default is 0.3)")
    args = parser.parse_args()

    predict_ori_mask_dir = os.path.join(args.output_folder, 'FullMasks')
    ori_mask_dir = os.path.join(args.output_folder, 'OrigMasks')

    process_pred_masks(args.pred_mask_folder, args.mask_folder, predict_ori_mask_dir, ori_mask_dir, args.increase_size)