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
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No mask found in the image.")
        return None
    main_contour = max(contours, key=cv2.contourArea)

    # Calculate the bounding box around the main contour
    x, y, w, h = cv2.boundingRect(main_contour)

    center_x = x + w // 2
    center_y = y + h // 2

    return center_x, center_y, h

def slide_location(image, center_y, slide_height):
    # Read the original image
    original_image = image

    # Calculate the top and bottom limits for cropping
    max_top = max(center_y - slide_height // 2, 0)
    max_bottom = min(center_y + slide_height // 2, original_image.shape[0])

    # Adjust center_y if it's too close to the top or bottom
    if max_top == 0:
        center_y = slide_height // 2
    elif max_bottom == original_image.shape[0]:
        center_y = original_image.shape[0] - slide_height // 2
    elif max_bottom == 0:
        center_y = slide_height // 2

    return center_y

def pad_bottom_to_height(image, target_height):
    current_height, width = image.shape[:2]
    if current_height >= target_height:
        return image
    pad_height = target_height - current_height
    bottom_pad = pad_height #- top_pad
    padded_image = cv2.copyMakeBorder(image, 0, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_image

def pad_top_to_height(image, target_height):
    current_height, width = image.shape[:2]
    if current_height >= target_height:
        return image
    pad_height = target_height - current_height
    top_pad = pad_height
    padded_image = cv2.copyMakeBorder(image, top_pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_image

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
    center_x, center_y, height = find_center(original_mask)

    slide_height = size
    center_y = slide_location(original_mask, center_y, slide_height)

    # Crop the slide window around the center
    original_image = cv2.imread(image_path)
    original_image  = resize_images(original_image , size)
    slide_window = original_image[center_y - slide_height // 2: center_y + slide_height // 2, :]


    # Save the resized slide window
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    #print(" image ", base_name)
    save_path = os.path.join(save_folder, f"images/{base_name}.png")
    cv2.imwrite(save_path, slide_window)

    # Save the mask
    mask_name = f"{base_name}.png"
    save_mask_path = os.path.join(save_folder, f"masks/{mask_name}")
    #shutil.copy(mask_path, save_mask_path)
    mask_window = original_mask[center_y - slide_height // 2: center_y + slide_height // 2, :]
    cv2.imwrite(save_mask_path, mask_window)

    start = center_y - slide_height // 2
    end   = center_y + slide_height // 2
    # Save additional slide windows and masks
    #adjusted_slide_height = int(height * 0.3)
    adjusted_slide_height = int(size * 0.3)
    #upper_slide_window = original_image[max(center_y - slide_height // 2 - adjusted_slide_height, 0): center_y - slide_height // 2 - adjusted_slide_height, :]
    #lower_slide_window = original_image[center_y + slide_height // 2 + adjusted_slide_height: min(center_y + slide_height // 2 + adjusted_slide_height, size), :]
    #upper_mask_window  = original_mask[max(center_y - slide_height // 2 - adjusted_slide_height, 0): center_y - slide_height // 2 - adjusted_slide_height, :]
    #lower_mask_window  = original_mask[center_y + slide_height // 2 + adjusted_slide_height : min(center_y + slide_height // 2 + adjusted_slide_height, size), :]

    upper_slide_window = original_image[max(start - adjusted_slide_height, 0): end - adjusted_slide_height, :]
    #lower_slide_window = original_image[start + adjusted_slide_height: min(end - adjusted_slide_height, size), :]
    lower_slide_window = original_image[start + adjusted_slide_height: min(end + adjusted_slide_height, original_image.shape[0]), :]


    upper_mask_window  = original_mask[max(start - adjusted_slide_height, 0): end - adjusted_slide_height, :]
    #lower_mask_window  = original_mask[start + adjusted_slide_height : min(end + adjusted_slide_height, size), :]
    lower_mask_window = original_mask[start + adjusted_slide_height: min(end + adjusted_slide_height, original_image.shape[0]), :]

    upper_slide_window= pad_top_to_height(upper_slide_window, size)
    upper_mask_window= pad_top_to_height(upper_mask_window, size)
    #lower_slide_window =pad_bottom_to_height(lower_slide_window, size)
    #lower_mask_window =pad_bottom_to_height(lower_mask_window, size)

    # Save adjusted slide windows and masks if they are not empty
    if upper_slide_window.shape[0] > 0:
        upper_save_image_path = os.path.join(save_folder, f"images/{base_name}_upper.png")
        upper_save_mask_path = os.path.join(save_folder, f"masks/{base_name}_upper.png")
        cv2.imwrite(upper_save_image_path, upper_slide_window)
        cv2.imwrite(upper_save_mask_path, upper_mask_window)

    if lower_slide_window.shape[0] < size + 1 and lower_slide_window.shape[0] >0  :
        lower_save_image_path = os.path.join(save_folder, f"images/{base_name}_lower.png")
        lower_save_mask_path = os.path.join(save_folder, f"masks/{base_name}_lower.png")
        cv2.imwrite(lower_save_image_path, lower_slide_window)
        cv2.imwrite(lower_save_mask_path, lower_mask_window)


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
