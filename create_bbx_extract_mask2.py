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

def crop_and_save(image, mask, bboxes, output_image_dir, output_mask_dir, base_filename, resize_to=None):
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        if resize_to:
            cropped_image = cv2.resize(cropped_image, resize_to)
            cropped_mask = cv2.resize(cropped_mask, resize_to)

        image_filename = os.path.join(output_image_dir, f"{base_filename}_crop_{i}.png")
        mask_filename = os.path.join(output_mask_dir, f"{base_filename}_crop_{i}.png")
        cv2.imwrite(image_filename, cropped_image)
        cv2.imwrite(mask_filename, cropped_mask)

def process_images_and_masks(image_folder, mask_folder, output_folder, resize_to, increase_size):
    output_image_dir = os.path.join(output_folder, 'images')
    output_mask_dir = os.path.join(output_folder, 'masks')

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

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
        adjusted_bboxes = adjust_bboxes(bboxes, image.shape, increase_size)

        # Crop and save images and masks
        base_filename = os.path.splitext(image_file)[0]
        crop_and_save(image, mask, adjusted_bboxes, output_image_dir, output_mask_dir, base_filename, resize_to)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop regions from images and masks based on bounding boxes")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the folder containing image files")
    parser.add_argument('--mask_folder', type=str, required=True, help="Path to the folder containing mask files")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save cropped images and masks")
    parser.add_argument('--resize_to', type=int, nargs=2, default=None, help="Resize cropped images and masks to this size (width height)")
    parser.add_argument('--increase_size', type=float, default=0.3, help="Percentage to increase the size of the bounding boxes (default is 0.3)")
    args = parser.parse_args()

    process_images_and_masks(args.image_folder, args.mask_folder, args.output_folder, tuple(args.resize_to) if args.resize_to else None, args.increase_size)