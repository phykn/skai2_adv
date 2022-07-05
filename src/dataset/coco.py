import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..util import open_image, save_json


def make_coco_file(
    image_folder: str,
    annot_file: str,
    dst: str
) -> None:
    # data
    df_data = pd.read_csv(annot_file)
    image_files = df_data["img_id"].unique()

    # initialize
    annt_id = 0
    images = []
    annotations = []
    categories = [
        dict(id=0, supercategory="background", name="none"),
        dict(id=1, supercategory="unknown", name="unknown"),
        dict(id=2, supercategory="normal", name="normal"),
        dict(id=3, supercategory="unscrewed", name="unscrewed_red"),
        dict(id=4, supercategory="rusty", name="rusty_yellow"),
        dict(id=5, supercategory="rusty", name="rusty_red"),
        dict(id=6, supercategory="unscrewed", name="unscrewed_yellow"),
    ]
    
    # images, annotations
    for file_name in tqdm(image_files):
        file, ext = os.path.splitext(file_name)
        image_id = int(file.split("_")[-1])
        
        I = open_image(os.path.join(image_folder, file_name))        
        image_width, image_height = I.size
        image = dict(
            id=image_id,
            file_name=file_name,
            height=image_height,
            width=image_width
        )

        annots = df_data.loc[df_data["img_id"]==file_name]
        _annotations = []
        for i in range(len(annots)):
            annot = annots.iloc[i]
            x1 = np.clip(annot["x1"]*image_width, 0, image_width)
            x2 = np.clip(annot["x2"]*image_width, 0, image_width)
            y1 = np.clip(annot["y1"]*image_height, 0, image_height)
            y2 = np.clip(annot["y2"]*image_height, 0, image_height)

            width = int(round(x2-x1))
            height = int(round(y2-y1))
            x1 = int(round(x1))
            x2 = int(round(x2))
            y1 = int(round(y1))
            y2 = int(round(y2))

            annotation = dict(
                id=annt_id,
                image_id=image_id,
                category_id=int(annot["class_id"]),
                is_crowd=0,
                area=width*height,
                bbox=[x1, y1, width, height],
                segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]]
            )
            _annotations.append(annotation)
            annt_id += 1

        images += [image]
        annotations += _annotations

    coco = dict(
        images=images,
        annotations=annotations,
        categories=categories
    )
    
    # save file
    save_json(coco, dst)
    print(f"File saved: {dst}")