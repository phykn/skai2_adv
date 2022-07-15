import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader 

from dataset.bbox import yolo_to_norm_xyxy
from dataset.dataset import Test_Dataset
from model.classifier import Twin_Classifier


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
    parser.add_argument("--root", default="inference", type=str, help="root folder")
    parser.add_argument("--output", default="submission.csv", type=str, help="output file name")
    parser.add_argument("--weight", default="weight.pt", type=str, help="model weight file")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="num workers")
    parser.add_argument("--pin_memory", default=True, type=str2bool, help="pin memory")    
    parser.add_argument("--base_file_len", default=14, type=int, help="base length of file name")
    return parser.parse_args()


def read_label(
    path: str
) -> pd.DataFrame:
    img_id = os.path.split(path)[1].replace(".txt", ".jpg")
    
    if os.path.exists(path):
        with open(path, "r") as f: 
            lines = f.readlines()
        
        df = []
        for line in lines:
            line = line.replace("\n", "")
            items = line.split(" ")
            items = [float(item) for item in items]
            bbox = yolo_to_norm_xyxy(items[1:5])

            df_row = dict(
                img_id=img_id,
                class_id=1,
                score=items[5],
                x1=bbox[0],
                y1=bbox[1],
                x2=bbox[2],
                y2=bbox[3]
            )
            df_row = pd.DataFrame(pd.Series(df_row)).T
            df.append(df_row)
        df = pd.concat(df)
        return df
    
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
        df = pd.DataFrame(pd.Series(df_row)).T
        return df
    
    
def main(args):
    # get bboxes
    img_paths = sorted(glob(os.path.join(args.img_folder, "*.jpg")))
    df = []
    for img_path in tqdm(img_paths, desc="convert text to dataframe"):
        img_id = os.path.split(img_path)[1]    
        label = os.path.join(args.root, "labels", img_id.replace(".jpg", ".txt"))
        df_label = read_label(label)
        df.append(df_label)
    df = pd.concat(df).reset_index(drop=True)

    # rename crop images
    crop_paths = sorted(glob(os.path.join(args.root, "crops", "*/*jpg")))
    for path in tqdm(crop_paths, desc="rename"):
        folder, file = os.path.split(path)
        name, ext = os.path.splitext(file)

        num = name[args.base_file_len:]
        num = int(num) if num!="" else 1
        name = f"{name[:args.base_file_len]}{num:04d}"

        os.rename(path, os.path.join(folder, f"{name}{ext}"))

    # get crop image paths
    crop_paths = sorted(glob(os.path.join(args.root, "crops", "*/*jpg")))
    
    # load model    
    model = Twin_Classifier()
    model.load_state_dict(torch.load(args.weight))
    model = model.clf
    model.cuda()
    model.eval()
    
    # dataset
    dataset = Test_Dataset(
        files=crop_paths,
        img_size=224
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory,
        shuffle=False, 
    )
    
    # predict
    class_ids = []
    scores = []
    for data in tqdm(dataloader, desc="predict"):
        data = {k: v.cuda() for k, v in data.items()}

        with torch.no_grad():
            pred = model(data["input_image"])
            pred = F.softmax(pred["logit"], dim=1).cpu().numpy()        

        class_id = np.argmax(pred, axis=1) + 2
        score = np.max(pred, axis=1)

        class_ids.append(class_id)
        scores.append(score)

    class_ids = np.concatenate(class_ids)
    scores = np.concatenate(scores)
    
    # submission file
    df_bolt = df.query("class_id==1").copy()
    df_bolt["class_id"] = class_ids
    df_bolt["score"] = df_bolt["score"].values * scores
    df_none = df.query("class_id==0").copy()
    df_result = pd.concat([df_bolt, df_none]).sort_index()
    
    # save file
    dst = os.path.join(args.root, args.output)
    df_result.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"file saved: {dst}")
    
    
if __name__ == "__main__":
    main(get_args())