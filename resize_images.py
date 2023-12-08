import os
import cv2
import argparse
from tqdm import tqdm

def resize_images(source_folder, destination_folder, target_size):
    os.makedirs(destination_folder, exist_ok=True)

    for root, dirs, files in os.walk(source_folder):
        for file in tqdm(files, desc="Processing images", unit="image"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                # Load the image
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)

                # Resize the image
                resized_img = cv2.resize(img, target_size)

                # Save the resized image
                destination_path = os.path.join(destination_folder, file)
                cv2.imwrite(destination_path, resized_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images in a folder and its subfolders.")
    parser.add_argument("--source_folder", required=True, help="Path to the source folder containing images.")
    parser.add_argument("--destination_folder", required=True, help="Path to the destination folder for resized images.")
    parser.add_argument("--target_size", default=(640, 640), type=int, nargs=2, help="Target size for resizing (width height).")

    args = parser.parse_args()

    # Convert target_size to a tuple
    target_size = tuple(args.target_size)

    # Resize images
    resize_images(args.source_folder, args.destination_folder, target_size)
