import numpy as np
import pandas as pd
from typing import List, Tuple


def resample_id(
    df: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    image_list = []
    image_nums = []
    for class_id in [0, 2, 3, 4, 5, 6]:
        df_class = df.loc[df["class_id"]==class_id]
        images = list(df_class["img_id"].unique())
        
        image_list.append(images)
        image_nums.append(len(images))

    weights = np.round(np.max(image_nums) / np.array(image_nums)).astype(int)
    weights[0] = 1

    images = []
    for _list, weight in zip(image_list, weights):
        for _ in range(weight): images += _list

    img_ids, repeats = np.unique(images, return_counts=True)
    return list(img_ids), list(repeats)