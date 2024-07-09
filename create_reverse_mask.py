import os
import cv2
import argparse
from tqdm import tqdm

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

def process_pred_masks(pred_mask_folder, mask_folder, predict_ori_mask_dir):
    os.makedirs(predict_ori_mask_dir, exist_ok=True)

    pred_masks = [f for f in os.listdir(pred_mask_folder) if f.endswith('.png') or f.endswith('.jpg')]

    for pred_mask_file in tqdm(pred_masks, desc="Processing predicted masks"):
        print(f"Processing {pred_mask_file}")
        base_filename = os.path.splitext(pred_mask_file)[0].rsplit('_', 1)[0]
        pred_mask_path = os.path.join(pred_mask_folder, pred_mask_file)
        mask_path = os.path.join(mask_folder, f"{base_filename}.png")

        if not os.path.exists(mask_path):
            print(f"Original mask {base_filename}.png not found.")
            continue

        # Read original mask and predicted mask
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

        if original_mask is None:
            print(f"Failed to read original mask {mask_path}")
            continue
        if pred_mask is None:
            print(f"Failed to read predicted mask {pred_mask_path}")
            continue

        # Find bounding boxes from the original mask
        bboxes = find_bboxes_from_mask(original_mask)

        # Adjust bounding boxes
        adjusted_bboxes = adjust_bboxes(bboxes, original_mask.shape)

        # Resize and paste predicted masks back to the original mask
        for i, bbox in enumerate(adjusted_bboxes):
            new_mask = resize_and_paste_pred_mask(original_mask, pred_mask, bbox)
            predict_ori_mask_path = os.path.join(predict_ori_mask_dir, f"{base_filename}_predictOriMask_{i}.png")
            cv2.imwrite(predict_ori_mask_path, new_mask)
            print(f"Saved {predict_ori_mask_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Resize and paste predicted masks back to the original masks")
    parser.add_argument('--pred_mask_folder', type=str, required=True, help="Path to the folder containing predicted masks")
    parser.add_argument('--mask_folder', type=str, required=True, help="Path to the folder containing original masks")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save the new masks with predicted masks pasted")
    args = parser.parse_args()

    process_pred_masks(args.pred_mask_folder, args.mask_folder, args.output_folder)
