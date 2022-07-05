import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import cv2


def proc_img(img_file, img_output_dir, suspect_dir, img_size: int = 256):
    """Process single image through pipeline"""
    f = Path(img_file)
    error = False
    exception = None
    res = pd.DataFrame(columns=['img', 'stddev', 'low_freq_prop', 'lap_var', 'exception', 'suspect'])

    # Tests if data format of image is correct
    try:
        img = cv2.imread(str(f), )
    except Exception as e:
        error = True
        exception = e
    if (os.stat(f).st_size == 0) or (img is None):
        error = True
        exception = "Zero byte size or empty image file."
    try:
        stddev = np.round(np.std(img), 5)
        low_freq_prop = len(img[img < 25]) / len(img.ravel())
        lap_var = np.var(cv2.Laplacian(img, cv2.CV_64F))
    except Exception as e:
        exception = f"Image is corrupt or otherwise unusable: {e}"
        error = True
    if error:
        shutil.copy(f, os.path.join(suspect_dir, f.name))
        print(f"An error occurred while trying to read the file: {exception}")
        res.loc[len(res)] = [suspect_dir / f.name, None, None, None, exception, error]
        return res

    # Tests on content of image. These aren't errors per se, but images that are not acceptable for other reasons
    # LROC images tend to appear corrupt or contain unnatural repetitive patterns
    if check_dim_ratio(img, 25):
        print("The dimensions of this image seem wrong. Moving to suspect dir.")
        exception = "Image distorted"
        print(f"Image {f} has dimensions suggesting distortions. Please investigate.")
        error = True
        exception = f"Distorted dimensions."
    if (low_freq_prop < .02) or (lap_var > 4000):
        print(f"Some properties of this image look suspect (noise, repetitive patterns, etc.), moving to suspect dir")
        error = True
        exception = f"Noise."
    if error:
        shutil.copy(f, os.path.join(suspect_dir, f.name))
        print(f"There is image quality issues with this file: {exception}.")
        res.loc[len(res)] = [suspect_dir / f.name, stddev, low_freq_prop, lap_var, exception, error]
        return res

    # If previous error detectors pass, we can now process
    print(f"Good file! Now image processing and exporting to file")
    img = scale2shortest(img, img_size=img_size)
    imgs = photobooth_cut(img)

    for i, imglet in enumerate(imgs):
        imglet_filename = f.name.split(sep='.')[0] + f'_{i}' + '.jpg'
        imglet_path = img_output_dir / imglet_filename
        stddev = np.std(imglet)
        low_freq_prop = len(imglet[imglet < 25]) / len(imglet.ravel())
        lap_var = np.var(cv2.Laplacian(imglet, cv2.CV_64F))
        # Now we need to rerun some checks in case a patch exhibits bad image characteristics
        if stddev < 10:
            cv2.imwrite(os.path.join(suspect_dir, imglet_filename), imglet)
            exception = "Image exhibiting stddev < 10."
            error = True
        else:
            cv2.imwrite(os.path.join(img_output_dir, imglet_filename), imglet)
        res.loc[len(res)] = [imglet_path, stddev, low_freq_prop, lap_var, exception, error]

    return res


def check_dim_ratio(img, threshold: float = 30):
    height, width, _ = img.shape
    if np.round(height / width) >= threshold:
        return 1
    return 0


def verify_image(img):
    """Open an image and run some basic tests to ensure the data is correct"""
    pass


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