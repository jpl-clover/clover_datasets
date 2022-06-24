import os
from pathlib import Path
import shutil
import sys

import numpy as np
import cv2
from PIL import Image, ImageStat

def process_imgs(img_parent_dir: Path, img_output_dir: Path, img_files):
    """Process list of images"""
    suspect_dir = img_output_dir / 'suspect'
    suspect_dir.mkdir(exist_ok=True)

    for f in img_files:
        img_filepath = os.path.join(img_parent_dir, f)
        # shutil.copy(os.path.join(img_parent_dir, f), os.path.join(img_output_dir, f))
        try:
            # img = Image.open(img_filepath)
            img = cv2.imread(img_filepath, )
            #img = np.flip(img, axis=-1)
        except IOError:
            print(f"This looks like a bad file potentially, moving to suspect dir")
            shutil.copy(os.path.join(img_parent_dir, f), os.path.join(suspect_dir, f))
            continue

        #try:
        #    img_stats = ImageStat.Stat(img)
        #except IOError:
        #    print(f"This looks like a bad file potentially, moving to suspect dir")
        #    shutil.copy(os.path.join(img_parent_dir, f), os.path.join(suspect_dir, f))
        #    continue

        # stddev = img_stats.stddev[0]
        try:
            stddev = np.std(img)
        except TypeError:
            print(f"This looks like a bad file potentially, moving to suspect dir")
            shutil.copy(os.path.join(img_parent_dir, f), os.path.join(suspect_dir, f))
            continue

        low_freq_prop = len(img[img < 25]) / len(img.ravel())
        if low_freq_prop < .02:
            print(f"This looks like a bad image, moving to suspect dir")
            print(f"StdDev: {stddev}, < 5 pixel intensity ratio: {low_freq_prop}")
            shutil.copy(os.path.join(img_parent_dir, f), os.path.join(suspect_dir, f))
            continue

        print(f"Good file!")
        shutil.copy(os.path.join(img_parent_dir, f), os.path.join(img_output_dir, f))
        print(f"StdDev: {stddev}, < 5 pixel intensity ratio: {low_freq_prop}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~")



def get_img_stddev(img):
    """Get standard deviation of image intensity"""
    pass

