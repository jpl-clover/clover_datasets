import argparse
from src.datasets import CLOVERDatasets


def get_args_parser():
    parser = argparse.ArgumentParser('CLOVERDatasets', add_help=False)

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--dataset_mslv2', action='store_true',
                               help="Mutually exclusive with other dataset_ arguments.")
    dataset_group.add_argument('--dataset_lroc', action='store_true',
                               help="Mutually exclusive with other dataset_ arguments.")

    parser.add_argument('--lroc_phase', type=int, choices=range(1, 29),
                        help="Mission phase denoted by value 1-28.")
    parser.add_argument('--lroc_dtype', type=str, default='edr',
                        help="LROC data type. Default of EDR is acceptable for most cases.")
    parser.add_argument('--lroc_num_patches', type=int,
                        help="Number of patches to sample from LROC image patches.")

    parser.add_argument('--num_images', type=int, help="Number of images in dataset.")
    parser.add_argument('--data_source', type=str, required=True,
                        default='~/clover_shared/datasets')
    parser.add_argument('--out_path', type=str, required=True,
                        default='~/datasets/clover_processed')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLOVERDatasets', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.dataset_lroc and ((args.num_images is None) or (args.lroc_phase is None)):
        parser.error("--dataset_lroc requires --num_images and lroc_phase to be defined.")

    print(f'\nInput data will come from: {args.data_source}')
    print(f'Output datasets will be made in: {args.out_path}')
    clover_datasets = CLOVERDatasets(data_path=args.data_source, out_path=args.out_path)

    if args.dataset_lroc:
        clover_datasets.create_lroc_dataset(lroc_phase=args.lroc_phase,
                                            num_images=args.num_images,
                                            patches=args.lroc_num_patches,
                                            lroc_dtype=args.lroc_dtype)
