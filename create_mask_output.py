import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def process_images_and_masks(image_folder, mask_folder, output_folder, increase_size):
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # List all images in the image folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Processing images"):
        # Read the image
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        # Convert the image to RGB (OpenCV loads images in BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Define the range for the specific purple color in RGB
        lower_purple = np.array([0, 0, 128])
        upper_purple = np.array([50, 50, 255])
        
        # Create a mask for the specific purple color
        mask_purple = cv2.inRange(image_rgb, lower_purple, upper_purple)
        
        # Find contours of the bounding box
        contours, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the corresponding mask file (same name but different extension)
        base_name = os.path.splitext(image_file)[0]
        mask_file = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_mask_file = base_name + ext
            if os.path.exists(os.path.join(mask_folder, potential_mask_file)):
                mask_file = potential_mask_file
                break

        if mask_file is None:
            print(f"Warning: Could not find corresponding mask for {image_file}")
            continue
        
        # Read the corresponding mask image
        mask_path = os.path.join(mask_folder, mask_file)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            print(f"Warning: Could not read mask {mask_path}")
            continue
        
        # Create a new mask with the same dimensions as the mask image
        new_mask = np.zeros_like(mask_image)
        
        for contour in contours:
            # Get the bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            
            # Increase the size of the bounding box
            x_incr = int(w * increase_size)
            y_incr = int(h * increase_size)
            x = max(0, x - x_incr)
            y = max(0, y - y_incr)
            w = min(mask_image.shape[1] - x, w + 2 * x_incr)
            h = min(mask_image.shape[0] - y, h + 2 * y_incr)
            
            # Apply the bounding box to the mask
            new_mask[y:y+h, x:x+w] = mask_image[y:y+h, x:x+w]
        
        # Save the new mask to the output folder
        output_mask_path = os.path.join(output_folder, base_name + '.png')
        cv2.imwrite(output_mask_path, new_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images to find bounding boxes and apply them to mask images")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the folder containing images")
    parser.add_argument('--mask_folder', type=str, required=True, help="Path to the folder containing masks")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save new masks")
    parser.add_argument('--increase_size', type=float, default=0.0, help="Increase size of bounding boxes detected (default is 0.0)")
    args = parser.parse_args()
    
    process_images_and_masks(args.image_folder, args.mask_folder, args.output_folder, args.increase_size)