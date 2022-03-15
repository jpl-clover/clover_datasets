import shutil
from pathlib import Path

import pandas as pd
from torchvision.datasets import ImageFolder


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
        self.df_msl_train = None

    def generate_mslv2_dataset(self, train_file: str = 'train-set-v2.1.txt',
                               msl_dataset_dir: str = 'msl_dataset'):
        """Create Pytorch image dataset format compatible directory structure"""
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
        print(f"MSLv2 training dataset summary:\n {value_counts}")