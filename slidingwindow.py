import os
import cv2
import argparse
from tqdm import tqdm
import shutil

import os
import cv2
import argparse
from tqdm import tqdm


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




def slideprocess(input_folder, save_folder, size, overlap):
    """Function to resize images while keeping the aspect ratio"""

    # Create save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Iterate over the files in the input folder
    for image_name in tqdm(os.listdir(input_folder), desc="Resizing images"):
        image_path = os.path.join(input_folder, image_name)
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        img_res = resize_images(image, size)

        # Original image dimensions
        image_height = img_res.shape[0]
        image_width = img_res.shape[1]

        # Calculate overlap pixels
        overlap_pixels = int(size * overlap)

        # Calculate the number of splits
        num_splits = (image_height - overlap_pixels) // (size - overlap_pixels) #+ 1
        #print("code:",num_splits, "--",overlap_pixels,"--", image_height, "xx", image_width )
        remain  =  (image_height - overlap_pixels) % (size - overlap_pixels)
        if remain >0.2: num_splits= num_splits+1

        #print("code:",num_splits, "--",overlap_pixels,"--", image_height, "xx", image_width )

        # Iterate over the splits
        for i in range(num_splits):
            # Calculate the starting and ending positions for each split
            start_y = i * (size - overlap_pixels)
            end_y = min(start_y + size, image_height)

            last_y = start_y + size
            if last_y > image_height:
                end_y = image_height
                start_y = image_height - size

            # Extract the split as a small image
            small_image = img_res[start_y:end_y, :]

            # Resize the small image
            #small_image = cv2.resize(small_image, (size, size))

            # Save the 
            i = i+1;
            save_image_path = os.path.join(save_folder, f"{image_name[:-4]}_{i}.png")
            cv2.imwrite(save_image_path, small_image)



        # Save resized image
        #save_path = os.path.join(save_folder, image_name)
        #cv2.imwrite(save_path, resized_image)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize images while keeping aspect ratio')
    parser.add_argument('--input_folder', required=True, help='Input folder containing images')
    parser.add_argument('--save_folder', required=True, help='Folder to save resized images')
    parser.add_argument('--overlap', type=float, default=0.3, help='Overlap ratio (default: 0.3)')
    parser.add_argument('--size', type=int, default=640, help='Size of each window (default: 640)')

    args = parser.parse_args()

    slideprocess(args.input_folder, args.save_folder, args.size, args.overlap)
