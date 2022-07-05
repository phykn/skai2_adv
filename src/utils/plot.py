import numpy as np
import matplotlib.pyplot as plt
from typing import List
from matplotlib.patches import Rectangle
from ..dataset.bbox import norm_xyxy_to_pixel_xyxy


def plot_bbox(
    image: np.array,
    bboxes: List[float],
) -> None:
    """box is normalized xyxy"""
    image_height, image_width, _ = image.shape

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(image)
    for bbox in bboxes:
        x1, y1, x2, y2 = norm_xyxy_to_pixel_xyxy(bbox, image_width, image_height)
        ax.add_patch(Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none"
        ))
    plt.show()