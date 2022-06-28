import os
from pathlib import Path
import shutil
import sys
import numpy as np
import cv2


def proc_img(img_file, img_output_dir, suspect_dir, img_size: int = 256):
    """Process single image through pipeline"""
    f = Path(img_file)
    error = False
    exception = None

    # Tests if data format of image is correct
    try:
        img = cv2.imread(str(f), )
    except Exception as e:
        print(f"Read image error occured: {e}")
        error = True
        exception = e
    if (os.stat(f).st_size == 0):
        print(f"This is a zero byte file, moving to suspect dir.")
        error = True
        exception = "Zero byte size file."
    if img is None:
        print(f"This does not appear to be an image file. Moving to suspect dir.")
        error = True
        exception = "Image produces empty object."
    try:
        stddev = np.round(np.std(img), 5)
        low_freq_prop = len(img[img < 25]) / len(img.ravel())
        lap_var = np.var(cv2.Laplacian(img, cv2.CV_64F))
        print(f"StdDev: {stddev}, low pixel intensity ratio: {low_freq_prop}, laplace variance: {lap_var}. ", end="")
    except Exception as e:
        print(f"Something is wrong with this image file: {e}")
        error = True
        exception = "Image is corrupt or otherwise unusable."
    if error:
        shutil.copy(f, os.path.join(suspect_dir, f.name))
        return 2

    # Tests on content of image. These aren't errors per se, but images that are not acceptable for other reasons
    if (low_freq_prop < .02) or (lap_var > 4000):
        print(f"Some properties of this image look suspect (noise, repetitive patterns, etc.), moving to suspect dir")
        shutil.copy(f, os.path.join(suspect_dir, f.name))
        return 1

    # If previous error detectors pass, we can now process
    print(f"Good file! Now image processing and exporting to file")
    img = scale2shortest(img, img_size=img_size)
    imgs = photobooth_cut(img)

    for i, imglet in enumerate(imgs):
        imglet_filename = f.name.split(sep='.')[0] + f'_{i}' + '.jpg'
        stddev = np.std(imglet)
        if stddev < 10:
            cv2.imwrite(os.path.join(suspect_dir, imglet_filename), imglet)
        else:
            cv2.imwrite(os.path.join(img_output_dir, imglet_filename), imglet)

    return 0


def normalize(img):
    pixels = img.astype('float32')
    pixels /= 255.0
    return pixels


def scale2shortest(img, img_size: int = 256):
    """Base on scale image to ratio of shortest side of image. Preserves aspect ratio """
    height, width, _ = img.shape
    scale = img_size / height

    if height > width:
        scale = img_size / width

    height = int(img.shape[0] * scale)
    width = int(img.shape[1] * scale)
    img_resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

    return img_resized


def photobooth_cut(img):
    """Cut an image short length size down long-side of image... like a photobooth

    Args:

    Returns:
        A list of images
    """
    length, width, _ = img.shape
    imgs = []

    if length == width:
        imgs.append(img)
        return(imgs)

    vertical = True
    if length < width:
        vertical = False

    if vertical:
        patches = int(length / width)
        for i in range(patches):
            img_s = img[(i * width):((i + 1) * width), :, :]
            imgs.append(img_s)
    else:
        patches = int(width / length)
        for j in range(patches):
            img_s = img[:, (j * length):((j + 1) * length), :]
            imgs.append(img_s)

    return imgs


def check_low_proportion_low_intensity(img, max_intensity: int = 25, threshold: float = .02):
    """Computes proportion of pixels that are below threshold value (0-255)

    Low intensity values correspond to dark areas which typically denote shadows, details, and other
    features common in most natural photos. Abnormal images consisting of a smooth gradient, uniform snow, or
    other unnatural images will tend to have a very low proportion of low intensity pixels.
    """
    low_freq_prop = len(img[img < max_intensity]) / len(img.ravel())