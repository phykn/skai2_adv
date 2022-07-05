import os
import argparse
import pandas as pd
from tqdm import tqdm
from dataset.bbox import norm_xyxy_to_yolo


def get_args_parser():
    parser = argparse.ArgumentParser("Make YOLO style label", add_help=False)
    parser.add_argument("--src_csv_path", default="data/train.csv", type=str, help="source csv path")
    parser.add_argument("--src_img_folder", default="data/train", type=str, help="source image folder")
    parser.add_argument("--file_column", default="img_name", type=str, help="column which contains file name")
    parser.add_argument("--single_object", action='store_true', help="all class id is 0")
    return parser


def main(args):
    df = pd.read_csv(args.src_csv_path)
    group = df.groupby(by=[args.file_column])
    for img_file in tqdm(df[args.file_column].unique(), desc="Labeling"):
        # set txt file path
        txt_file = os.path.join(args.src_img_folder, f"{os.path.splitext(img_file)[0]}.txt")

        # process
        df_tmp = group.get_group(img_file)
        for i in range(len(df_tmp)):
            data = df_tmp.iloc[i]

            class_id = data["class_id"]
            x_center, y_center, width, height = norm_xyxy_to_yolo(
                [data["x1"], data["y1"], data["x2"], data["y2"]]
            )

            if class_id != 0:
                # set class
                if args.single_object:
                    class_id = 0
                else:
                    class_id = class_id - 2

                # write file
                if i == 0:
                    with open(txt_file, "w") as f:
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                else:
                    with open(txt_file, "a") as f:
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Make YOLO style label", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)