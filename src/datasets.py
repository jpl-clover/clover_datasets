from pathlib import Path
from torchvision.datasets import ImageFolder


class CloverDatasets(object):

    def __init__(self, datapath: str = 'clover_shared/datasets',
                 outpath: str = 'datasets/clover_datasets'):
        self.datapath = Path(datapath)