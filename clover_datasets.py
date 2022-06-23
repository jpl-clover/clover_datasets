import argparse
from src.datasets import CLOVERDatasets


def get_args_parser():
    parser = argparse.ArgumentParser('CLOVERDatasets', add_help=False)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--dataset_mslv2', action='store_true',
                               help="Mutually exclusive with other dataset_ arguments.")
    dataset_group.add_argument('--dataset_lroc', action='store_true',
                               help="Mutually exclusive with other dataset_ arguments.")
    parser.add_argument('--num_images', type=int, help="Number of images in dataset.")
    parser.add_argument('--data_source', type=str, required=True,
                        default='~/clover_shared/datasets')
    parser.add_argument('--out_path', type=str, required=True,
                        default='~/datasets/clover_processed')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLOVERDatasets', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.dataset_lroc and (args.num_images is None):
        parser.error("--dataset_lroc requires --num_images to be used.")

    print(f'\nInput data will come from: {args.data_source}')
    print(f'Output datasets will be made in: {args.out_path}')
    clover_datasets = CLOVERDatasets(data_path=args.data_source, out_path=args.out_path)

    if args.dataset_lroc:
        clover_datasets.create_lroc_dataset()
