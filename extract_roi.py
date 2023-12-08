#
# Mamary
# Update 8 Dec 2023
# Tommy bugs
# Miscellaneous utilities.

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
from tqdm import tqdm
import pydicom
from ultralytics import YOLO

class extract_ROI():
    def __init__(self, df, yolo_model):
        self.df = df
        self.detect_model = yolo_model
        self.folder = None
        self.df_loc = None
        self.count_access = 0
        self.count_error = 0

    def __len__(self):
        return len(self.df)

    def load_image(self, idx=0):
        """
        Method to load the image.

        Parameters:
            - idx (int or str): index or path of the image.

        Returns:
            - img (numpy.ndarray): image as an 8-bit 3-channel array.
        """
        path = idx
        if isinstance(idx, int):
            self.df_loc = self.df.iloc[idx]
            path = self.df_loc.Path
        else:
            self.df_loc = self.df[self.df['Path'] == idx]

        mode = path.split('.')[-1]
        img = None

        if mode == 'dcm':
            ds = pydicom.dcmread(path)
            img2d = ds.pixel_array
            voi_lut = apply_voi_lut(img2d, ds)
            if np.sum(voi_lut) > 0:
                img2d = voi_lut
            img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min())
            img2d = (img2d * 255).astype(np.uint8)

            if ds.PhotometricInterpretation == 'MONOCHROME1':
                img2d = np.invert(img2d)

            img = cv2.cvtColor(img2d, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.imread(path)

        return img

    def crop(self, img):
        """
        Method to crop the image based on YOLO object detection.
        If YOLO fails to find any boxes, it falls back to using crop_threshold with extract_roi_otsu.

        Parameters:
            - img (numpy.ndarray): Input image.

        Returns:
            - img (numpy.ndarray): Cropped image.
        """
        results = self.detect_model(img)
        boxes = results[0].boxes

        if len(boxes) == 0:
            print("YOLO did not detect any boxes. Falling back to crop_threshold with extract_roi_otsu.")
            img = self.crop_threshold(img)
        else:
            # Assuming there's only one box, you might need to modify this if there are multiple boxes
            box = boxes[0]
            xy = box.xyxy
            x1 = int(xy[0][0].item())
            y1 = int(xy[0][1].item())
            x2 = int(xy[0][2].item())
            y2 = int(xy[0][3].item())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[y1: y2, x1:x2]

        return img

    def extract_roi_otsu(self, img, gkernel=(5, 5)):
        """
        Extracts the Region of Interest (ROI) from an input image using Otsu's thresholding.

        Parameters:
            - img: Input image (numpy array).
            - gkernel: Size of the Gaussian kernel for image smoothing. Default is (5, 5).

        Returns:
            - roi_coords: Coordinates (x0, y0, x1, y1) of the extracted ROI.
            - area_percentage: Area percentage of the ROI relative to the input image.
            - additional_info: Additional information (currently set to None).
        """
        ori_h, ori_w = img.shape[:2]

        upper = np.percentile(img, 95)
        img[img > upper] = np.min(img)

        if gkernel is not None:
            img = cv2.GaussianBlur(img, gkernel, 0)

        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
        img_bin = cv2.dilate(img_bin, element)

        cnts, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) == 0:
            return None, None, None

        areas = np.array([cv2.contourArea(cnt) for cnt in cnts])
        select_idx = np.argmax(areas)
        cnt = cnts[select_idx]
        area_pct = areas[select_idx] / (img.shape[0] * img.shape[1])

        x0, y0, w, h = cv2.boundingRect(cnt)

        x1 = min(max(int(x0 + w), 0), ori_w)
        y1 = min(max(int(y0 + h), 0), ori_h)
        x0 = min(max(int(x0), 0), ori_w)
        y0 = min(max(int(y0), 0), ori_h)

        return [x0, y0, x1, y1], area_pct, None

    def crop_threshold(self, img):
        """
        Crop the image based on thresholding with extract_roi_otsu.

        Parameters:
            - img (numpy.ndarray): Input image.

        Returns:
            - img (numpy.ndarray): Cropped image.
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply extract_roi_otsu to get the ROI coordinates
        roi_coords, _, _ = self.extract_roi_otsu(img_gray)

        if roi_coords is not None:
            img = img_gray[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
        else:
            print("Extracting ROI with Otsu thresholding and contour detection failed. Returning original image.")

        return img

    def process_and_save_single_image_with_mask(self, input_image_path, input_mask_path, output_image_path, output_mask_path):
        """
        Process a single image and its corresponding mask, and save the processed image and mask.

        Parameters:
            - input_image_path (str): Path to the input image.
            - input_mask_path (str): Path to the corresponding mask.
            - output_image_path (str): Path to save the processed image.
            - output_mask_path (str): Path to save the processed mask.
        """
        img = self.load_image(input_image_path)
        mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)

        # Process image and mask
        processed_img = self.crop(img)
        processed_mask = self.crop_threshold(mask)

        # Save processed image and mask
        cv2.imwrite(output_image_path, processed_img)
        cv2.imwrite(output_mask_path, processed_mask)

        print(f"Processed image saved to: {output_image_path}")
        print(f"Processed mask saved to: {output_mask_path}")


