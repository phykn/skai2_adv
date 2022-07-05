import numpy as np
from typing import List


def norm_xyxy_to_pixel_xyxy(
    bbox: List[float],
    image_width: int,
    image_height: int
) -> List[int]:
    x1, y1, x2, y2 = bbox
    x1 = np.clip(np.round(x1*image_width), 0, image_width)
    x2 = np.clip(np.round(x2*image_width), 0, image_width)
    y1 = np.clip(np.round(y1*image_height), 0, image_height)
    y2 = np.clip(np.round(y2*image_height), 0, image_height)
    return [int(x1), int(y1), int(x2), int(y2)]


def norm_xyxy_to_yolo(
    bbox: List[float]
) -> List[float]:
    x1, y1, x2, y2 = bbox
    x_center = np.clip((x1+x2)/2, 0, 1)
    y_center = np.clip((y1+y2)/2, 0, 1)
    width = np.clip(x2-x1, 0, 1)
    height = np.clip(y2-y1, 0, 1)
    return [x_center, y_center, width, height]
