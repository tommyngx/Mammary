import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def truncate_normalize(img):
    img_min = np.min(img)
    img_max = np.max(img)
    normalized_img = (img - img_min) / (img_max - img_min)
    return normalized_img

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab_planes[0] = clahe.apply(lab_planes[0])
    clahe_img = cv2.merge(lab_planes)
    return cv2.cvtColor(clahe_img, cv2.COLOR_LAB2BGR)

def apply_musica(img, alpha=2.5, beta=1.5, pyramid_levels=4):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_pyramid = [img_gray]

    for i in range(pyramid_levels):
        img_gray = cv2.pyrDown(img_gray)
        gaussian_pyramid.append(img_gray)

    laplacian_pyramid = [gaussian_pyramid[-1]]

    for i in range(pyramid_levels, 0, -1):
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], cv2.pyrUp(gaussian_pyramid[i]))
        laplacian_pyramid.append(laplacian)

    musica_pyramid = []

    for laplacian, gauss in zip(laplacian_pyramid, reversed(gaussian_pyramid[:-1])):
        enhanced_layer = cv2.addWeighted(laplacian, alpha, gauss, beta, 0)
        musica_pyramid.append(enhanced_layer)

    enhanced_img = np.sum(musica_pyramid, axis=0)
    return cv2.cvtColor(enhanced_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def enhance_images(source_folder, destination_folder, styles):
    for root, dirs, files in os.walk(source_folder):
        relative_path = os.path.relpath(root, source_folder)
        destination_path = os.path.join(destination_folder, relative_path)

        for file in tqdm(files, desc=f"Enhancing images in {relative_path}", unit="image"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                source_image_path = os.path.join(root, file)
                destination_image_path = os.path.join(destination_path, file)

                os.makedirs(destination_path, exist_ok=True)

                # Load the image
                img = cv2.imread(source_image_path)

                # Apply selected enhancement styles
                for style in styles:
                    if style == 'truncate_normalize':
                        img = truncate_normalize(img)
                    elif style == 'clahe':
                        img = apply_clahe(img)
                    elif style == 'musica':
                        img = apply_musica(img)

                # Save the enhanced image
                cv2.imwrite(destination_image_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance images with various styles.")
    parser.add_argument("--source_folder", required=True, help="Path to the source folder containing images.")
    parser.add_argument("--destination_folder", required=True, help="Path to the destination folder for enhanced images.")
    parser.add_argument("--styles", nargs='+', default=['truncate_normalize'], choices=['truncate_normalize', 'clahe', 'musica'], help="Enhancement styles to apply.")

    args = parser.parse_args()

    # Enhance images
    enhance_images(args.source_folder, args.destination_folder, args.styles)
