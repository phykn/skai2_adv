import os
import argparse
import yaml
import pandas as pd
from typing import List
from sklearn.model_selection import KFold


def get_args_parser():
    parser = argparse.ArgumentParser("Split Data", add_help=False)
    parser.add_argument("--src_csv_path", default="data/train.csv", type=str, help="source csv path")
    parser.add_argument("--src_img_folder", default="data/train", type=str, help="source image folder")
    parser.add_argument("--dst_txt_folder", default="data/split", type=str, help="destination txt folder")
    parser.add_argument("--n_splits", default=5, type=int, help="number of fold")
    parser.add_argument("--single_object", action='store_true', help="one class")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    return parser


def save_yaml(
    path: str,
    train: str,
    valid: str,
    nc: int,
    names: List[str],
    verbose: bool=False
) -> None:
    data = dict(
        train=train,
        val=valid,
        nc=nc,
        names=names
    )
    with open(path, "w") as f:
        yaml.dump(data, f)

    if verbose:
        print(f"Created: {path}")


def write_data(
    path: str,
    data: List[str]
) -> None:
    with open(path, "w") as f:
        for item in data:
            f.write(f"{item}\n")


def main(args):
    df = pd.read_csv(args.src_csv_path)
    img_ids = df["img_id"].unique()

    kf = KFold(
        n_splits=args.n_splits,
        random_state=args.seed,
        shuffle=True
    )
    for i, (train_index, valid_index) in enumerate(kf.split(img_ids)):
        # set txt path
        train_txt_path = os.path.join(args.dst_txt_folder, f"train_{i:02d}.txt")
        valid_txt_path = os.path.join(args.dst_txt_folder, f"valid_{i:02d}.txt")

        # select data
        df_train = df.query(f"img_id=={list(img_ids[train_index])}")
        df_valid = df.query(f"img_id=={list(img_ids[valid_index])}")

        train_image_paths = [os.path.join(args.src_img_folder, file) for file in df_train["img_name"].unique()]
        valid_image_paths = [os.path.join(args.src_img_folder, file) for file in df_valid["img_name"].unique()]

        # write
        write_data(train_txt_path, train_image_paths)
        write_data(valid_txt_path, valid_image_paths)

        # save yaml file
        if args.single_object:
            save_yaml(
                path=os.path.join(args.dst_txt_folder, f"dataset_{i:02d}.yaml"),
                train=train_txt_path,
                valid=valid_txt_path,
                nc=1,
                names=["bolt"],
                verbose=True
            )
        else:
            save_yaml(
                path=os.path.join(args.dst_txt_folder, f"dataset_{i:02d}.yaml"),
                train=train_txt_path,
                valid=valid_txt_path,
                nc=5,
                names=["normal", "unscrewed_red", "rusty_yellow", "rusty_red", "unscrewed_yellow"],
                verbose=True
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Split Data", parents=[get_args_parser()])
    args = parser.parse_args()

    # make dir
    os.makedirs(args.dst_txt_folder, exist_ok=True)

    # main
    main(args)