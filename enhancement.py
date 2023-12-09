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

def apply_musica(img):
    def resize_image(img):
        return cv2.resize(img, (img.shape[1] & -2, img.shape[0] & -2))

    def laplacian_pyramid(img, L):
        gauss = gaussian_pyramid(img, L)
        lp = []
        for layer in range(L):
            tmp = cv2.pyrUp(gauss[layer + 1], dstsize=(gauss[layer].shape[1], gauss[layer].shape[0]))[:gauss[layer].shape[0], :gauss[layer].shape[1]]
            tmp_channels = tmp.shape[2]
            gauss_layer_channels = gauss[layer][:, :, :tmp_channels]
            tmp = gauss_layer_channels - tmp
            lp.append(tmp)
        lp.append(gauss[L][:, :, :3])
        return lp

    def gaussian_pyramid(img, L):
        tmp = resize_image(img)
        gp = [tmp]
        for layer in range(L):
            tmp = cv2.pyrDown(tmp)
            gp.append(tmp)
        return gp

    L = 3  # You can adjust this parameter as needed
    params = {'M': 50, 'p': 0.5, 'a': [1, 1, 1]}  # You can adjust these parameters as needed

    # Create Laplacian pyramid
    lp = laplacian_pyramid(img, L)

    # Enhance coefficients
    M = params['M']
    p = params['p']
    a = params['a']
    for layer in range(L):
        x = lp[layer]
        x[x < 0] = 0.0
        G = a[layer] * M
        enhanced_layer = G * np.multiply(np.divide(x, np.abs(x), out=np.zeros_like(x), where=x != 0),
                                          np.power(np.divide(np.abs(x), M), p))
        # Clip values to the valid range [0, 255]
        lp[layer] = np.clip(enhanced_layer, 0, 255).astype(np.uint8)


    # Reconstruct image
    rs = resize_image(img)
    for i in range(L - 1, -1, -1):
        expanded = pyramid_expand(rs, upscale=2, multichannel=True, preserve_range=True)
        rs = expanded + lp[i]

    return rs[:img.shape[0], :img.shape[1]]




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
