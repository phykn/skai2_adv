import os
import argparse
import zipfile


def get_args_parser():
    parser = argparse.ArgumentParser("Unzip file", add_help=False)
    parser.add_argument('-file', type=str)
    parser.add_argument('-dst', type=str)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unzip file", parents=[get_args_parser()])
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    with zipfile.ZipFile(args.file, "r") as file:
        file.extractall(args.dst)
    print(f"DONE: {args.file}")