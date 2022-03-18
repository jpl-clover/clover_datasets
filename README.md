# CLOVER Datasets
## Basic Documentation
### Create a MSLV2 Dataset
Commands all run in folder containing this repo.

#### Command Line
```shell
python src/generate_clover_datasets.py --gen_mslv2 [train_data_list.txt] --data_source [CLOVER dataset rootdir]
--out_path [directory where propoerly organized data goes]
```
Example. Generate MSLv2 dataset, use `train-set-v2.1.txt` list, and `clover_shared/datasets` as root data directory
which will output to `$SCRATCH` in a directory called `datasets/mslv2_dataset`
```shell
python src/generate_clover_datasets.py --gen_mslv2 ~/clover_shared/datasets/msl-labeled-data-set-v2.1/train-set-v2.1.txt --data_source ~/clover_shared/datasets/ --out_path $SCRATCH/datasets/mslv2_dataset/
```
#### Code
```python
from datasets import CLOVERDatasets

my_dataset = CLOVERDatasets(data_path='~/clover_shared/datasets', 
                            out_path='$SCRATCH/datasets/CLOVER_processed/')
my_dataset.gen_mslv2_dataset(train_file='~/clover_shared/datasets/msl-labeled-data-set-v2.1/train-set-v2.1.txt',
                             msl_dataset_dir='mslv2_dataset')
```
