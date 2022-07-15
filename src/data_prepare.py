import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from dataset.resample import resample_id
from dataset.open_image import open_image
from dataset.image_process import (
    delete_objects, extract_objects, transform_bbox, transform_s, transform_m, transform_l
)


def get_args_parser():
    parser = argparse.ArgumentParser("Data preparation module", add_help=False)
    parser.add_argument("--src_csv_path", default="data/train.csv", type=str, help="source csv path")
    parser.add_argument("--src_img_folder", default="data/train", type=str, help="source image folder")
    parser.add_argument("--dst_csv_path", default="data_prepared/train.csv", type=str, help="destination prepared data csv path")
    parser.add_argument("--dst_img_folder", default="data_prepared/image", type=str, help="destination prepared image folder")
    parser.add_argument("--img_size", default=640, type=int, help="image size")
    parser.add_argument("--repeat_l", default=2, type=int, help="number of repeat for large transform")
    parser.add_argument("--repeat_m", default=4, type=int, help="number of repeat for medium transform")
    parser.add_argument("--repeat_s", default=8, type=int, help="number of repeat for small transform")
    return parser


def data_row(
    img_id: str,
    class_ids: np.ndarray,
    bboxes: np.ndarray,
    img_name: str
) -> dict:
    return pd.DataFrame(dict(
        img_id=[img_id]*len(class_ids),
        class_id=list(class_ids),
        x1=list(bboxes[:, 0]),
        y1=list(bboxes[:, 1]),
        x2=list(bboxes[:, 2]),
        y2=list(bboxes[:, 3]),
        img_name=[img_name]*len(class_ids)
    ))


def main(args):
    df = pd.read_csv(args.src_csv_path)

    df_out = []
    for img_id, repeat in zip(*resample_id(df)):
        print(f"Source Image: {img_id}")

        # get file name
        file, ext = os.path.splitext(img_id)

        # open image
        image = open_image(os.path.join(args.src_img_folder, img_id))

        # select img id
        df_img_id = df.loc[df["img_id"]==img_id]
        bboxes = df_img_id[["x1", "y1", "x2", "y2"]].values
        class_labels = df_img_id["class_id"].values

        # original image
        origin_file_name = f"{file}_origin{ext}"
        Image.fromarray(image).save(
            os.path.join(args.dst_img_folder, origin_file_name)
        )
        df_out.append(data_row(img_id, class_labels, bboxes, origin_file_name))

        # no bbox
        if class_labels[0] == 0:
            # transform_l
            for i in range(repeat*args.repeat_l):
                trans_file_name = f"{file}_trans_l_{i:03d}{ext}"
                output = transform_bbox(transform_l(img_size=args.img_size), image, bboxes, class_labels)
                Image.fromarray(output["image"]).save(
                    os.path.join(args.dst_img_folder, trans_file_name)
                )
                df_out.append(data_row(img_id, output["class_labels"], output["bboxes"], trans_file_name))

        else:
            # delete objects
            no_obj_file_name = f"{file}_no_obj{ext}"
            no_obj_image = delete_objects(image, bboxes)
            Image.fromarray(no_obj_image).save(
                os.path.join(args.dst_img_folder, no_obj_file_name)
            )
            df_out.append(data_row(img_id, np.array([0]), np.array([[0, 0, 0, 0]]), no_obj_file_name))

            # transform_l
            for i in range(repeat*args.repeat_l):
                trans_file_name = f"{file}_trans_l_{i:03d}{ext}"
                output = transform_bbox(transform_l(img_size=args.img_size), image, bboxes, class_labels)
                Image.fromarray(output["image"]).save(
                    os.path.join(args.dst_img_folder, trans_file_name)
                )
                df_out.append(data_row(img_id, output["class_labels"], output["bboxes"], trans_file_name))

            # transform_m
            for i in range(repeat*args.repeat_m):
                trans_file_name = f"{file}_trans_m_{i:03d}{ext}"
                output = transform_bbox(transform_m(img_size=args.img_size), image, bboxes, class_labels)
                Image.fromarray(output["image"]).save(
                    os.path.join(args.dst_img_folder, trans_file_name)
                )
                df_out.append(data_row(img_id, output["class_labels"], output["bboxes"], trans_file_name))

            # transform_s
            for i in range(repeat*args.repeat_s):
                trans_file_name = f"{file}_trans_s_{i:03d}{ext}"
                output = transform_bbox(transform_s(img_size=args.img_size), image, bboxes, class_labels)
                Image.fromarray(output["image"]).save(
                    os.path.join(args.dst_img_folder, trans_file_name)
                )
                df_out.append(data_row(img_id, output["class_labels"], output["bboxes"], trans_file_name))

    # save csv file
    df_out = pd.concat(df_out).reset_index(drop=True)
    df_out.to_csv(args.dst_csv_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data preparation module", parents=[get_args_parser()])
    args = parser.parse_args()
    
    # make dir
    os.makedirs(args.dst_img_folder, exist_ok=True)

    # main
    main(args)