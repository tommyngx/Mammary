import os, shutil
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def find_center(original_mask):
    # Read the mask image
    #mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask  = cv2.cvtColor(original_mask, cv2.COLOR_BGR2GRAY)
    if mask is None:
        print("Failed to read the mask image.")
        return None

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No mask found in the image.")
        return None

    # Get the largest contour (assuming it's the main mask)
    main_contour = max(contours, key=cv2.contourArea)

    # Calculate the bounding box around the main contour
    x, y, w, h = cv2.boundingRect(main_contour)

    # Calculate the center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    return center_x, center_y

def slide_location(image, center_y, slide_height):
    # Read the original image
    original_image = image

    # Calculate the top and bottom limits for cropping
    max_top = min(center_y - slide_height // 2, 0)
    max_bottom = min(center_y + slide_height // 2, original_image.shape[0])

    # Adjust center_y if it's too close to the top or bottom
    if max_top == 0:
        center_y = slide_height // 2
    elif max_bottom == original_image.shape[0]:
        center_y = original_image.shape[0] - slide_height // 2

    return center_y


def resize_images(image, size):
    # Get original image dimensions
    height, width = image.shape[:2]

    # Calculate aspect ratio
    aspect_ratio = width / height

    # Calculate new dimensions based on specified width
    new_width = size
    new_height = int(new_width / aspect_ratio)

    # Resize image while keeping aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


def crop_slide_window(image_path, mask_path, save_folder, size):
    original_mask = cv2.imread(mask_path)
    original_mask = resize_images(original_mask , size)
    # Find the center of the mask
    center = find_center(original_mask)
    if center is None:
        return

    # Determine the slide location
    center_x, center_y = center
    slide_height = size
    center_y = slide_location(original_mask, center_y, slide_height)

    # Crop the slide window around the center
    original_image = cv2.imread(image_path)
    original_image  = resize_images(original_image , size)
    slide_window = original_image[center_y - slide_height // 2: center_y + slide_height // 2, :]


    # Save the resized slide window
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_folder, f"images/{base_name}.png")
    cv2.imwrite(save_path, slide_window)

    # Save the mask
    mask_name = f"{base_name}.png"
    save_mask_path = os.path.join(save_folder, f"masks/{mask_name}")
    #shutil.copy(mask_path, save_mask_path)
    mask_window = original_mask[center_y - slide_height // 2: center_y + slide_height // 2, :]
    print(" image ", basename)
    cv2.imwrite(save_mask_path, mask_window)

def process_images(input_folder, save_folder, size):
    # Create the save folder if it doesn't exist
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.makedirs(os.path.join(save_folder, 'images'))
        os.makedirs(os.path.join(save_folder, 'masks'))

    # Iterate over the images in the input folder
    input_image_folder = os.path.join(input_folder, 'images')
    input_mask_folder = os.path.join(input_folder, 'masks')

    for filename in tqdm(os.listdir(input_image_folder), desc="Processing images"):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_image_folder, filename)
            mask_path = os.path.join(input_mask_folder, filename)
            if not os.path.exists(mask_path):
                print(f"Mask not found for {filename}. Skipping...")
                continue
            crop_slide_window(image_path, mask_path, save_folder, size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop slide windows around mask centers and resize them.')
    parser.add_argument('--input_folder', required=True, help='Input folder containing images and masks')
    parser.add_argument('--save_folder', required=True, help='Folder to save cropped and resized slide windows')
    parser.add_argument('--size', type=int, default=448, help='Height of the resized slide windows (default: 448)')
    args = parser.parse_args()

    process_images(args.input_folder, args.save_folder, args.size)
