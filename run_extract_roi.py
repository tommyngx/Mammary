import os
import argparse
import cv2
import pandas as pd
from tqdm import tqdm
import pydicom
from extract_roi import extract_ROI  # Assuming you have the extract_ROI class in a separate file
from ultralytics import YOLO

def main(args):
    # Load YOLO model
    yolo_model = YOLO(args.yolo_model_path)

    # Automatically create a DataFrame from the images folder
    df = create_dataframe(args.images_folder,args.masks_folder )

    # Initialize extract_ROI object
    extractor = extract_ROI(df, yolo_model)

    # Provide input paths and output folder
    images_folder = args.images_folder
    masks_folder = args.masks_folder
    output_folder = args.output_folder

    # Process and save images and masks
    extractor.process_and_save_images_with_masks(images_folder, masks_folder, output_folder)
    #extractor.plot_sample(resize=256)  # Adjust the method based on available functionality

def load_yolo_model(model_path):
    yolo8 = YOLO(model_path)
    return yolo8  

def create_dataframe(images_folder, masks_folder):
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm'))]
    df = pd.DataFrame({"Path": [os.path.join(images_folder, img) for img in image_files]})
    df['ID'] = df['Path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df['Path_mask'] = df['ID'].apply(lambda x: os.path.join(masks_folder, f"{x}.png"))  # Adjust the mask file extension if needed
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run extract_ROI on images and masks.")
    parser.add_argument("--yolo_model_path", required=True, help="Path to the YOLO model.")
    parser.add_argument("--images_folder", required=True, help="Path to the folder containing images.")
    parser.add_argument("--masks_folder", required=True, help="Path to the folder containing masks.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for processed images and masks.")

    args = parser.parse_args()
    main(args)