import os
import json
import cv2
import argparse
from tqdm import tqdm
import numpy as np

def process_images(json_file, image_folder, output_folder):
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create a dictionary mapping image IDs to image file names
    image_id_to_filename = {image['id']: image['file_name'] for image in data['images']}
    
    # Create a dictionary mapping image IDs to their bounding boxes
    image_id_to_bboxes = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        if image_id not in image_id_to_bboxes:
            image_id_to_bboxes[image_id] = []
        image_id_to_bboxes[image_id].append(bbox)
    
    # Process each image
    for image_id, filename in tqdm(image_id_to_filename.items(), desc="Processing images"):
        # Read the image
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        # Create a mask with the same dimensions as the image
        mask = np.zeros_like(image)
        
        # Keep the area of the bounding boxes
        for bbox in image_id_to_bboxes.get(id, []):
            x, y, w, h = map(int, bbox)
            mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]
        
        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images to keep only areas within bounding boxes")
    parser.add_argument('--json_file', type=str, required=True, help="Path to the COCO format JSON file")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the folder containing images")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save processed images")
    args = parser.parse_args()
    
    process_images(args.json_file, args.image_folder, args.output_folder)