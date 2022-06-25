import os
from pathlib import Path
import shutil
import sys

import numpy as np
import cv2


def process_imgs(img_parent_dir: Path, img_output_dir: Path, img_files):
    """Process list of images"""
    suspect_dir = img_output_dir / 'suspect'
    suspect_dir.mkdir(exist_ok=True)

    for f in img_files:
        img_filepath = os.path.join(img_parent_dir, f)

        try:
            img = cv2.imread(img_filepath, )
        except IOError:
            print(f"This looks like a bad file potentially, moving to suspect dir")
            shutil.copy(os.path.join(img_parent_dir, f), os.path.join(suspect_dir, f))
            continue

        try:
            stddev = np.round(np.std(img), 5)
        except TypeError:
            print(f"This looks like a bad file potentially, moving to suspect dir")
            shutil.copy(os.path.join(img_parent_dir, f), os.path.join(suspect_dir, f))
            continue

        low_freq_prop = len(img[img < 25]) / len(img.ravel())
        lap_var = np.var(cv2.Laplacian(img, cv2.CV_64F))

        if low_freq_prop < .02:
            print(f"This looks like a bad image, moving to suspect dir")
            print(f"StdDev: {stddev}, < 5 pixel intensity ratio: {low_freq_prop}")
            shutil.copy(os.path.join(img_parent_dir, f), os.path.join(suspect_dir, f))
            continue

        # Lines, repetitive patterns, snow
        if lap_var > 4000:
            print(f"This looks like repetitive pattern and may be bad image, moving to suspect dir")
            print(f"StdDev: {stddev}, < 5 pixel intensity ratio: {low_freq_prop}")
            shutil.copy(os.path.join(img_parent_dir, f), os.path.join(suspect_dir, f))
            continue

        print(f"Good file!")
        shutil.copy(os.path.join(img_parent_dir, f), os.path.join(img_output_dir, f))
        print(f"StdDev: {stddev}, < 5 pixel intensity ratio: {low_freq_prop}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

def check_low_proportion_low_intensity(img, max_intensity: = 25, proportion_threshold: float = .02):
    """Computes proportion of pixels that are below threshold value (0-255)

    Low intensity values correspond to dark areas which typically denote shadows, details, and other
    features common in most natural photos. Abnormal images consisting of a smooth gradient, uniform snow, or
    other unnatural images will tend to have a very low proportion of low intensity pixels.
    """
    low_freq_prop = len(img[img < max_intensity]) / len(img.ravel())