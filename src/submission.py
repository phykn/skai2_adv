import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import List

import torch
from torch.utils.data import DataLoader

from dataset.bbox import yolo_to_norm_xyxy
from dataset.dataset import Test_Dataset
from model.classifier import Classifier
from utils.nms import non_max_suppression_index


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser("Submission Module", add_help=False)
    parser.add_argument("--img_folder", default="data/predict", type=str, help="image folder")
    parser.add_argument("--inference_folder", default="inference", type=str, help="inference_folder")
    parser.add_argument("--output", default="submission.csv", type=str, help="output file name")

    parser.add_argument("--use_clf", default=True, type=str2bool, help="use classification model")
    parser.add_argument("--clf_num_class", default=6, type=int, help="model label number")
    parser.add_argument("--clf_weight", default="weight.pt", type=str, help="model weight")
    parser.add_argument("--overlapThresh", default=0.9, type=float, help="overlap threshold for NMS")

    parser.add_argument("--img_size", default=224, type=int, help="img size")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="num workers")
    parser.add_argument("--cuda", default=True, type=str2bool, help="use gpu")
    return parser.parse_args()


label_dict = {
    0: "normal",
    1: "unscrewed_red",
    2: "rusty_yellow",
    3: "rusty_red",
    4: "unscrewed_yellow"
}


def read_label(
    path: str
) -> pd.DataFrame:
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]

    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()

        df = []
        memory = {i:0 for i in range(len(label_dict))}
        for line in lines:
            line = line.replace("\n", "")
            items = line.split(" ")
            items = [float(item) for item in items]
            class_id = int(items[0])
            bbox = yolo_to_norm_xyxy(items[1:5])
            score = items[5]
            memory[class_id] += 1

            df_row = dict(
                img_id=f"{name}.jpg",
                class_id=class_id+2,
                score=score,
                x1=bbox[0],
                y1=bbox[1],
                x2=bbox[2],
                y2=bbox[3],
                crop=f"{label_dict[class_id]}/{name}{memory[class_id]}.jpg" if memory[class_id] > 1 else f"{label_dict[class_id]}/{name}.jpg"
            )
            df_row = pd.DataFrame(pd.Series(df_row)).T
            df.append(df_row)
        df = pd.concat(df)
        return df

    else:
        df_row = dict(
            img_id=f"{name}.jpg",
            class_id=0,
            score=0,
            x1=0,
            y1=0,
            x2=0,
            y2=0,
            crop=None
        )
        df = pd.DataFrame(pd.Series(df_row)).T
        return df


def softmax(
    x: np.ndarray
) -> np.ndarray:
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def correction_bbox_zero_class_id(
    x: pd.Series
) -> pd.Series:
    class_id = x["class_id"]
    if class_id == 0:
        x["score"] = 0
        x["x1"] = 0
        x["y1"] = 0
        x["x2"] = 0
        x["y2"] = 0
    return x


def score_ensemble(x):
    class_num = 7

    class_id = x["class_id"]
    pos_score = x["score"]
    neg_score = (1 - pos_score)/5

    index = list(range(class_num))
    index.remove(1)
    index.remove(class_id)

    obd_prob = np.zeros(class_num)
    obd_prob[class_id] = pos_score
    obd_prob[index] = neg_score
    obd_prob = obd_prob / np.sum(obd_prob)

    clf_prob = x[["0", "1", "2", "3", "4", "5", "6"]].values
    clf_prob = clf_prob.astype(float)

    prob = np.sqrt(obd_prob*clf_prob)
    prob = prob / (np.sum(prob)+1e-7)

    x = x.copy()
    x["ensemble_class_id"] = np.argmax(prob)
    x["ensemble_score"] = np.max(prob)
    return x


def add_zero_class_id(
    df: pd.DataFrame,
    img_ids: List[str]
) -> pd.DataFrame:
    group = df.groupby("img_id")
    group_keys = list(group.groups.keys())
    df_out = []
    for img_id in img_ids:
        if img_id in group_keys:
            df_out.append(group.get_group(img_id))
        else:
            df_row = dict(
                img_id=img_id,
                class_id=0,
                score=0,
                x1=0,
                y1=0,
                x2=0,
                y2=0
            )
            df_row = pd.DataFrame(pd.Series(df_row)).T
            df_out.append(df_row)
    return pd.concat(df_out)


def NMS(
    df: pd.DataFrame,
    overlapThresh: float=0.9
) -> pd.DataFrame:
    img_ids = df["img_id"].unique()
    group = df.groupby(by="img_id")

    df_out = []
    for img_id in img_ids:
        df_tmp = group.get_group(img_id).reset_index(drop=True)
        index = non_max_suppression_index(
            bboxes=df_tmp[["x1", "y1", "x2", "y2"]].values,
            probs=df_tmp["score"].values,
            overlapThresh=overlapThresh
        )
        df_out.append(df_tmp.iloc[index])

    df_out = pd.concat(df_out, axis=0).reset_index(drop=True)
    return df_out


def post_process(
    df: pd.DataFrame,
    img_ids: List[str],
    use_clf: bool,
    overlapThresh: float=0.9
) -> pd.DataFrame:
    df = df.drop(columns=["crop"])
    if use_clf:
        df = df.apply(lambda x: score_ensemble(x), axis=1)
        df["class_id"] = df["ensemble_class_id"].values.astype(int)
        df["score"] = df["ensemble_score"]
        df = df.drop(
            columns=["clf_class_id", "clf_score", "0", "1", "2", "3", "4", "5", "6", "ensemble_class_id", "ensemble_score"]
        )
        df = NMS(df, overlapThresh=overlapThresh)

    df = df.apply(lambda x: correction_bbox_zero_class_id(x), axis=1)
    df = df.loc[df["class_id"] > 0]
    df = add_zero_class_id(df, img_ids)
    df = df.fillna(0)
    return df


def main(args):
    img_paths = sorted(glob(os.path.join(args.img_folder, "*.jpg")))
    df = []
    for img_path in tqdm(img_paths, desc="concatenate"):
        img_id = os.path.split(img_path)[1]
        name = os.path.splitext(img_id)[0]
        label_path = os.path.join(args.inference_folder, "labels", f"{name}.txt")
        df_label = read_label(label_path)
        df.append(df_label)
    df = pd.concat(df).reset_index(drop=True)

    if args.use_clf:
        clf_label = np.zeros(len(df))
        clf_score = np.zeros(len(df))
        clf_proba = np.zeros((len(df), 7))

        # get crop images
        crop_index = df["crop"].dropna().index.to_list()
        crop_paths = df["crop"].dropna().values
        crop_paths = [f"{args.inference_folder}/crops/{sub_path}" for sub_path in crop_paths]

        # dataset
        dataset = Test_Dataset(
            files=crop_paths,
            img_size=args.img_size
        )

        # dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory= True if args.cuda else False,
            shuffle=False,
        )

        # model
        model = Classifier(num_class=args.clf_num_class)
        model.load_state_dict(
            torch.load(args.clf_weight)
        )
        model.eval()
        if args.cuda:
            model.cuda()

        # predict
        pred_label = []
        pred_score = []
        pred_proba = []
        for data in tqdm(dataloader, desc="clf predict"):
            if args.cuda:
                data = {k: v.cuda() for k, v in data.items()}

            with torch.no_grad():
                pred = model.predict(data)

            pred = pred.detach().cpu().numpy()
            if args.clf_num_class == 6:
                pred[:, [0, 1, 2, 3, 4, 5]] = pred[:, [5, 0, 1, 2, 3, 4]]
                pred = np.insert(pred, 1, -np.inf, axis=1)

            proba = softmax(pred)
            label = np.argmax(proba, axis=1)

            pred_score.append(np.max(proba, axis=1))
            pred_label.append(label)
            pred_proba.append(proba)

        pred_label = np.concatenate(pred_label)
        pred_score = np.concatenate(pred_score)
        pred_proba = np.concatenate(pred_proba)

        clf_label[crop_index] = pred_label
        clf_score[crop_index] = pred_score
        clf_proba[crop_index] = pred_proba

        df["clf_class_id"] = clf_label.astype(int)
        df["clf_score"] = clf_score
        df_proba = pd.DataFrame(data=clf_proba, columns=[str(i) for i in range(7)])
        df = pd.concat([df, df_proba], axis=1)


    # save submission file
    img_ids = [os.path.split(path)[1] for path in img_paths]
    df_submission = post_process(df, img_ids, use_clf=args.use_clf, overlapThresh=args.overlapThresh)

    dst = os.path.join(args.inference_folder, args.output)
    df_submission.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"file saved: {dst}")


if __name__ == "__main__":
    main(get_args())