import os
import sys
import multiprocessing as mp
import shutil
from itertools import repeat
from pathlib import Path

import pandas as pd
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

import src.img_tools as img_tools


class CLOVERDatasets(object):

    def __init__(self, data_path: str = 'clover_shared/datasets',
                 out_path: str = 'datasets/clover_datasets',
                 msl_class_map: str = None):
        """Create datasets for use in Pytorch based SSL Models from CLOVER datasets

        """
        print(f"Setting up CloverDatasets with {data_path} as input source and "
              f"{out_path} for output.\n")
        self.data_path = Path(data_path)
        self.out_path = Path(out_path)
        self.msl_class_map = msl_class_map

        self.lroc_imgs = None
        self.df_msl_train = None
        self.dataset = None

    def create_mslv2_dataset(self, train_file: str = 'train-set-v2.1.txt', msl_dataset_dir: str = 'mslv2_dataset',
                             create_pt_dataset: bool = False, pt_dataset_xforms: transforms = None):
        """Generate folder structure for MSLv2 dataset and PyTorch DataSet

        train_file: text file that has one column of image names, and another column of classes
        msl_dataset_dir: subdirectory of out_path defined in object creation, so out_path/msl_dataset_dir
        create_pt_dataset: generate a PyTorch dataset via ImageFolder with pt_dataset_xforms applied.

        """
        train_path = self.out_path / msl_dataset_dir / 'train'
        self.df_msl_train = pd.read_csv(train_file, sep='\s', names=['img', 'label'])

        print(f"Deleting existing dataset at {self.out_path / msl_dataset_dir}\n")
        if train_path.exists() and train_path.is_dir():
            shutil.rmtree(train_path)
        train_path.mkdir(parents=True, exist_ok=True)

        for i in self.df_msl_train.label.unique():
            (train_path / f'class_{i}').mkdir()
        for i, row in self.df_msl_train.iterrows():
            shutil.copy(self.data_path / f'msl-labeled-data-set-v2.1/images/{row.img}',
                        self.out_path / train_path / f'class_{row.label}')
        value_counts = self.df_msl_train['label'].value_counts()

        if create_pt_dataset:
            print("Generating PyTorch dataset from images.")
            self.dataset = ImageFolder(self.out_path, transform=pt_dataset_xforms)

        print(f"MSLv2 training dataset summary:\n {value_counts}")

    def create_lroc_dataset(self, num_images: int = 1000, img_size: int = 256,
                            lroc_phase: int = 1, lroc_dtype: str = "edr"):
        """Create unlabeled training dataset from LROC images

        Directory structure of LROC mount basically follows convention:
        lroc_dtype (edr/cdr/rdr) / lroc_phase (1-28) / YYYDOY / [images...]

        """
        procs = mp.cpu_count()

        lroc_phase_str = str(lroc_phase)
        if lroc_phase < 10:
            lroc_phase_str = "0" + lroc_phase_str

        img_dir = self.data_path / lroc_dtype / f'lrolrc_00{lroc_phase_str}' / 'extras/browse'
        img_subdirs = os.scandir(img_dir)
        img_output_path = self.out_path / lroc_dtype / f'lrolrc_00{lroc_phase_str}'
        img_output_path.mkdir(parents=True, exist_ok=True)

        print(f"Generating LROC dataset from {self.data_path}/{lroc_dtype}/lrolrc_00{lroc_phase_str}")
        print(f"\nProcessing all files in {img_dir} and outputting to {self.out_path} while maintaining dir structure")

        try:
            pool = mp.Pool(processes=procs)

            for subdir in img_subdirs:
                if subdir.is_dir():
                   subdir_output_path = Path(img_output_path / Path(subdir.path).stem)
                   subdir_output_path.mkdir(exist_ok=True)
                   suspect_dir = subdir_output_path / 'suspect'
                   suspect_dir.mkdir(exist_ok=True)
                   img_files = [os.path.join(subdir.path, f) for f in os.listdir(subdir.path)
                                if os.path.isfile(os.path.join(subdir.path, f))]
                else:
                    continue
                # Multiprocess image processing
                pool.starmap(img_tools.proc_img,
                             zip(img_files, repeat(subdir_output_path), repeat(suspect_dir), repeat(img_size)))
        finally:
            pool.close()
            pool.join()

        print("Done.")

    def describe(self):
        """Provide useful information about datasets"""
        pass
