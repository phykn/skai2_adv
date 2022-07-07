import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dataset.open_image import open_image
from dataset.image_process import extract_objects


def get_args_parser():
    parser = argparse.ArgumentParser("Crop image for classification", add_help=False)
    parser.add_argument("--src_csv_path", default="data/train.csv", type=str, help="source csv path")
    parser.add_argument("--src_img_folder", default="data/train", type=str, help="source image folder")
    parser.add_argument("--dst_csv_path", default="data_prepared/train_crop.csv", type=str, help="destination prepared data csv path")
    parser.add_argument("--dst_img_folder", default="data_prepared/image_crop", type=str, help="destination prepared image folder")
    parser.add_argument("--scale", default=1.0, type=float, help="crop scale, should be larger than 1.")
    return parser


def main(args):
    df = pd.read_csv(args.src_csv_path)
    group = df.groupby(by=["img_id"])
    img_ids = df["img_id"].unique()

    new_img_id = 0
    df_out = []
    for img_id in tqdm(img_ids):
        df_img = group.get_group(img_id)

        extractor = extract_objects(
            image=open_image(os.path.join(args.src_img_folder, img_id)),
            bboxes=df_img[["x1", "y1", "x2", "y2"]].values,
            class_labels=df_img["class_id"].values,
            scale=args.scale
        )
        for image_crop, class_label in extractor:
            if class_label != 0:
                file_name = f"{new_img_id:04d}.jpg"

                Image.fromarray(image_crop).save(
                    os.path.join(args.dst_img_folder, file_name)
                )

                df_out.append(
                    pd.DataFrame(
                        {
                            "file": [file_name],
                            "label": [class_label]
                        }
                    )
                )
                new_img_id += 1

    df_out = pd.concat(df_out).reset_index(drop=True)
    df_out.to_csv(args.dst_csv_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Crop image for classification", parents=[get_args_parser()])
    args = parser.parse_args()

    # make dir
    os.makedirs(args.dst_img_folder, exist_ok=True)

    # main
    main(args)