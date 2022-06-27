import os
from pathlib import Path
import multiprocessing as mp
import shutil
import sys

import numpy as np
import cv2


def process_imgs(img_files, img_output_dir: Path):
    """Process list of images"""
    suspect_dir = img_output_dir / 'suspect'
    suspect_dir.mkdir(exist_ok=True)

    for f in img_files:
        f = Path(f)
        print("--------------------")
        print(f)

        try:
            img = cv2.imread(str(f), )
        except IOError:
            print(f"This looks like a bad or corrupt file, moving to suspect dir")
            shutil.copy(f, os.path.join(suspect_dir, f.name))
            continue
        if (os.stat(f).st_size == 0):
            print(f"This is a zero byte file, moving to suspect dir.")
            shutil.copy(f, os.path.join(suspect_dir, f.name))
            continue
        if img is None:
            print(f"This does not appear to be an image file. Moving to suspect dir.")
            shutil.copy(f, os.path.join(suspect_dir, f.name))
            continue

        try:
            stddev = np.round(np.std(img), 5)
        except IOError:
            print(f"Something is wrong with this image file, moving to suspect dir")
            shutil.copy(f, os.path.join(suspect_dir, f.name))
            continue
        except Exception as e:
            print(e)
            shutil.copy(f, os.path.join(suspect_dir, f.name))
            continue

        low_freq_prop = len(img[img < 25]) / len(img.ravel())
        lap_var = np.var(cv2.Laplacian(img, cv2.CV_64F))

        print(f"StdDev: {stddev}, low pixel intensity ratio: {low_freq_prop}, laplace variance: {lap_var}. ", end="")

        if (low_freq_prop < .02) or (lap_var > 4000):
            print(f"Some properties of this image look suspect(noise, etc.), moving to suspect dir")
            shutil.copy(f, os.path.join(suspect_dir, f.name))
            continue
        else:
            print(f"Good file!")
        shutil.copy(f, os.path.join(img_output_dir, f.name))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~")


def proc_img(img_file, img_output_dir, suspect_dir):
    f = Path(img_file)
    error = False
    exception = None

    print(f"--------------------\n{f}")

    try:
        img = cv2.imread(str(f), )
    except Exception as e:
        print(f"Read image error occured: {e}")
        shutil.copy(f, os.path.join(suspect_dir, f.name))
        error = True
        exception = e
    if (os.stat(f).st_size == 0):
        print(f"This is a zero byte file, moving to suspect dir.")
        shutil.copy(f, os.path.join(suspect_dir, f.name))
        error = True
        exception = "Zero byte size file."
    if img is None:
        print(f"This does not appear to be an image file. Moving to suspect dir.")
        shutil.copy(f, os.path.join(suspect_dir, f.name))
        error = True
        exception = "Image produces empty object."
    try:
        stddev = np.round(np.std(img), 5)
        low_freq_prop = len(img[img < 25]) / len(img.ravel())
        lap_var = np.var(cv2.Laplacian(img, cv2.CV_64F))
        print(f"StdDev: {stddev}, low pixel intensity ratio: {low_freq_prop}, laplace variance: {lap_var}. ", end="")
    except Exception as e:
        print(f"Something is wrong with this image file: {e}")
        shutil.copy(f, os.path.join(suspect_dir, f.name))
        error = True
        exception = "Image is corrupt or otherwise unusable."
    if error:
        return 2

    # Tests on content of image. These aren't errors per se, but images that are not acceptable for other reasons
    if (low_freq_prop < .02) or (lap_var > 4000):
        print(f"Some properties of this image look suspect (noise, repetitive patterns, etc.), moving to suspect dir")
        shutil.copy(f, os.path.join(suspect_dir, f.name))
        return 1
    else:
        print(f"Good file!")
        shutil.copy(f, os.path.join(img_output_dir, f.name))
        return 0
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

def mp_imgs(img_files, img_output_dir: Path):
    pass


def check_low_proportion_low_intensity(img, max_intensity: int = 25, threshold: float = .02):
    """Computes proportion of pixels that are below threshold value (0-255)

    Low intensity values correspond to dark areas which typically denote shadows, details, and other
    features common in most natural photos. Abnormal images consisting of a smooth gradient, uniform snow, or
    other unnatural images will tend to have a very low proportion of low intensity pixels.
    """
    low_freq_prop = len(img[img < max_intensity]) / len(img.ravel())