from pathlib import Path
from torchvision.datasets import ImageFolder


class CloverDatasets(object):

    def __init__(self, datapath: str = 'clover_shared/datasets',
                 outpath: str = 'datasets/clover_datasets'):
        self.datapath = Path(datapath)

    def generate_MSL_v2(self):
        """Create Pytorch image dataset format commpatible directory structure"""





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