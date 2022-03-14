import shutil
from pathlib import Path
from torchvision.datasets import ImageFolder


class CloverDatasets(object):

    def __init__(self, data_path: str = 'clover_shared/datasets',
                 out_path: str = 'datasets/clover_datasets',
                 msl_class_map: str = 'msl_synset_words-indexed.txt'):
        self.data_path = Path(data_path)
        self.out_path = Path(out_path)
        self.msl_class_map = msl_class_map

    def generate_mslv2_dataset(self, msl_dataset_dir: str = 'msl_dataset'):
        """Create Pytorch image dataset format compatible directory structure"""
        train_path = self.out_path / msl_dataset_dir / 'train'
        print(f"Deleting existing dataset at {self.out_path / msl_dataset_dir}")
        if train_path.exists() and train_path.is_dir():
            shutil.rmtree(train_path)
        train_path.mkdir(parents=True, exist_ok=True)
        # Create directory structure
        for i in range(25):
            print(train_path / f'class_{i}')
            (train_path / f'class_{i}').mkdir()








def pct_train(BASE_IMG_PATH, TRAIN_LABELS,
              BASE_DIR=Path('/home/kaipak/datasets/msl-v2.1-ssl-runs')):
    dest_path = BASE_DIR / 'train'
    print(dest_path)
    if dest_path.exists() and dest_path.is_dir():
        shutil.rmtree(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)
    print(dest_path)

    # Make directories for all images
    for label in LABEL_DESC.label_desc:
        (dest_path / label).mkdir()

    for i, row in TRAIN_LABELS.iterrows():
        shutil.copy(BASE_IMG_PATH / row.img, dest_path / row.label_desc)