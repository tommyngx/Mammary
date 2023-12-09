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

import cv2
import copy
import numpy as np
from skimage.transform import pyramid_reduce, pyramid_expand


def isPowerofTwo(x):
    return x and (not (x & (x - 1)))


def findNextPowerOf2(n):
    n = n - 1
    while n & n - 1:
        n = n & n - 1
    return n << 1


def resize_image(img):
    row, col = img.shape
    if isPowerofTwo(row):
        rowdiff = 0
    else:
        nextpower = findNextPowerOf2(row)
        rowdiff = nextpower - row

    if isPowerofTwo(col):
        coldiff = 0
    else:
        nextpower = findNextPowerOf2(col)
        coldiff = nextpower - col

    img_ = np.pad(img, ((0, rowdiff), (0, coldiff)), 'reflect')
    return img_


def gaussian_pyramid(img, L):
    tmp = copy.deepcopy(img)
    gp = [tmp]
    for layer in range(L):
        tmp = pyramid_reduce(tmp, preserve_range=True)
        gp.append(tmp)
    return gp


def laplacian_pyramid(img, L):
    gauss = gaussian_pyramid(img, L)
    lp = []
    for layer in range(L):
        tmp = pyramid_expand(gauss[layer + 1], preserve_range=True)
        tmp_channels = tmp.shape[2]
        
        # Ensure the number of channels is consistent
        gauss_layer_channels = gauss[layer][:, :, :tmp_channels]
        
        tmp = gauss_layer_channels - tmp
        lp.append(tmp)
    lp.append(gauss[L])
    return lp


def enhance_coefficients(laplacian, L, params):
    M = params['M']
    p = params['p']
    a = params['a']
    for layer in range(L):
        x = laplacian[layer]
        x[x < 0] = 0.0
        G = a[layer] * M
        laplacian[layer] = G * np.multiply(np.divide(x, np.abs(x), out=np.zeros_like(x), where=x != 0),
                                           np.power(np.divide(np.abs(x), M), p))
    return laplacian


def reconstruct_image(laplacian, L):
    rs = laplacian[L]
    for i in range(L - 1, -1, -1):
        rs = pyramid_expand(rs, preserve_range=True)
        rs = np.add(rs, laplacian[i])
    return rs


def apply_musica(img, L=3, params=None):
    if params is None:
        params = {'M': 50, 'p': 0.5, 'a': [1, 1, 1]}

    img_resized = resize_image(img)
    lp = laplacian_pyramid(img_resized, L)
    lp = enhance_coefficients(lp, L, params)
    rs = reconstruct_image(lp, L)
    rs = rs[:img.shape[0], :img.shape[1]]
    return rs


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
