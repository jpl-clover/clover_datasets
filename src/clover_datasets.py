import argparse
from datasets import CLOVERDatasets


def get_args_parser():
    parser = argparse.ArgumentParser('CLOVERDatasets', add_help=False)
    parser.add_argument('--generate_mslv2_dataset', type=str,
                        help="Path to file must be provided if this option is chosen.")
    parser.add_argument('--data_source', type=str, required=True,
                        default='~/clover_shared/datasets')
    parser.add_argument('--out_path', type=str, required=True,
                        default='~/datasets/clover_processed')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLOVERDatasets', parents=[get_args_parser()])
    args = parser.parse_args()
    print(f'\nInput data will come from: {args.data_source}')
    print(f'Output datasets will be made in: {args.out_path}')
    clover_datasets = CLOVERDatasets(data_path=args.data_source, out_path=args.out_path)

    if args.generate_mslv2_dataset:
        clover_datasets.generate_mslv2_dataset(args.generate_mslv2_dataset)
