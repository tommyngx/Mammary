#
# Mamary
# Update 8 Dec 2023
# Tommy bugs
# Miscellaneous utilities.
import cv2
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from IPython import display
import pydicom
import pandas as pd
from ultralytics import YOLO

class extract_ROI:
    def __init__(self, df, yolo_model):
        self.df = df
        self.detect_model = yolo_model
        self.folder = None
        self.df_loc = None
        self.count_access = 0
        self.count_error = 0

    def load_image(self, idx=0):
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
        results = self.detect_model(img, verbose=False, boxes=False)
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

    def crop_mask(self, img, mask):
        results = self.detect_model(img)
        boxes = results[0].boxes

        if len(boxes) == 0:
            print("YOLO did not detect any boxes. Falling back to crop_threshold with extract_roi_otsu.")            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi_coords, _, _ = self.extract_roi_otsu(img_gray)

            if roi_coords is not None:
                mask = mask[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
            else:
                print("Extracting ROI with Otsu thresholding and contour detection failed. Returning original image.")

            return mask

        else:
                # Assuming there's only one box, you might need to modify this if there are multiple boxes
                box = boxes[0]
                xy = box.xyxy
                x1 = int(xy[0][0].item())
                y1 = int(xy[0][1].item())
                x2 = int(xy[0][2].item())
                y2 = int(xy[0][3].item())
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = mask[y1: y2, x1:x2]

                return mask

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

    def process_and_save_images_with_masks(self, images_folder, masks_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            img_path  = row['Path']
            mask_path = row['Path_mask']
            img = self.load_image(img_path)
            mask = self.load_image(mask_path)

            # Use your YOLO model for object detection (assuming YOLO has a method like detect)
            results = self.detect_model(img)

            # Add your logic to process results and masks here
            # Example: cropped_img = self.crop(img)
            cropped_img = self.crop(img)
            cropped_mask = self.crop_mask(img, mask)

            images_subfolder = os.path.join(output_folder, "images")
            masks_subfolder = os.path.join(output_folder, "masks")
            os.makedirs(images_subfolder, exist_ok=True)
            os.makedirs(masks_subfolder, exist_ok=True)            
            # Save the processed image
            output_img_path = os.path.join(images_subfolder ,f"{os.path.basename(img_path)}")
            output_mask_path = os.path.join(masks_subfolder ,f"{os.path.basename(mask_path)}")
            #print(output_img_path)
            cv2.imwrite(output_img_path, cropped_img)
            cv2.imwrite(output_mask_path, cropped_mask)

    def plot_sample(self, resize=256):
        # Add your plot_sample logic here
        pass

# Example usage:
if __name__ == "__main__":
    # Example usage of YOLO model
    yolo_model = YOLO("/path/to/your/yolo/model")

    # Example DataFrame creation
    df = pd.DataFrame({"Path": ["/path/to/images/folder/image1.jpg", "/path/to/images/folder/image2.jpg"]})

    # Initialize extract_ROI object
    extractor = extract_ROI(df, yolo_model)

    # Example processing and saving
    images_folder = "/path/to/images/folder"
    masks_folder = "/path/to/masks/folder"
    output_folder = "/path/to/output/folder"

    extractor.process_and_save_images_with_masks(images_folder, masks_folder, output_folder)
    # extractor.plot_sample(resize=256)  # Adjust the method based on available functionality
