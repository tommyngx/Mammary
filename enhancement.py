import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import copy
from skimage.transform import pyramid_reduce, pyramid_expand

def truncate_normalize(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

def apply_clahe(img):
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge channels
    enhanced_img = cv2.merge([cl, a, b])

    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)

    return enhanced_img

def apply_retinex(img):
    # Convert image to float32 for accurate calculations
    img_float = img.astype(np.float32)

    # Logarithmic transform to enhance the dynamic range
    img_log = np.log1p(img_float)

    # Apply MSRCR
    sigma_list = [15, 80, 250]
    retinex = np.zeros_like(img_float)
    for sigma in sigma_list:
        img_blur = cv2.GaussianBlur(img_log, (0, 0), sigma)
        retinex += img_log - img_blur

    # Scale the enhanced image to the range [0, 255]
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255

    # Convert back to uint8
    retinex = retinex.astype(np.uint8)

    return retinex

def apply_unsharp(img, sigma=1.0, strength=1.5):
    # Create a blurred version of the original image
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)

    # Subtract the blurred image from the original
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

    # Clip pixel values to the valid range [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened


def enhance_images(input_folder, output_folder, styles):
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for file in tqdm(files, desc="Processing images", unit="image"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                input_image_path = os.path.join(root, file)
                output_image_path = os.path.join(output_folder, file)

                # Load the image
                img = cv2.imread(input_image_path)

                # Convert to 0-255 scale
                img = truncate_normalize(img)

                # Apply selected enhancement styles
                for style in styles:
                    if style == 'clahe':
                        img = apply_clahe(img)
                    elif style == 'retinex':
                        img = apply_retinex(img)
                    elif style == 'unsharp':
                        img = apply_unsharp(img) 

                # Save the enhanced image
                cv2.imwrite(output_image_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply image enhancement techniques.")
    parser.add_argument("--input_folder", required=True, help="Path to the input folder containing images.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for enhanced images.")
    parser.add_argument("--styles", nargs='+', choices=['truncate_normalize', 'clahe', 'retinex', 'unsharp'], default=['truncate_normalize'], help="Enhancement styles to apply.")

    args = parser.parse_args()

    # Enhance images
    enhance_images(args.input_folder, args.output_folder, args.styles)
