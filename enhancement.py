import os
import cv2
import argparse
from tqdm import tqdm

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

def apply_musica(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian pyramid
    pyramids = [gray]
    for i in range(5):
        gray = cv2.pyrDown(gray)
        pyramids.append(gray)

    # Amplify contrast using Laplacian pyramid
    enhanced_img = pyramids[-1]
    for i in range(4, 0, -1):
        expanded = cv2.pyrUp(enhanced_img)
        enhanced_img = cv2.addWeighted(pyramids[i], 2, expanded, -1, 0)

    return enhanced_img

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
                    elif style == 'musica':
                        img = apply_musica(img)
                    # Add more styles as needed

                # Save the enhanced image
                cv2.imwrite(output_image_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply image enhancement techniques.")
    parser.add_argument("--input_folder", required=True, help="Path to the input folder containing images.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for enhanced images.")
    parser.add_argument("--styles", nargs='+', choices=['truncate_normalize', 'clahe', 'musica'], default=['truncate_normalize'], help="Enhancement styles to apply.")

    args = parser.parse_args()

    # Enhance images
    enhance_images(args.input_folder, args.output_folder, args.styles)
