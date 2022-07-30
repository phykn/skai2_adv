import numpy as np
import matplotlib.pyplot as plt
from typing import List
from matplotlib.patches import Rectangle
from ..dataset.bbox import norm_xyxy_to_pixel_xyxy


text_dict = {
    0: "bg",
    1: "etc",
    2: "normal",
    3: "unscrewed_red",
    4: "rusty_yellow",
    5: "rusty_red",
    6: "unscrewed_yellow"
}

color_dict = {
    0: "gray",
    1: "black",
    2: "green",
    3: "blue",
    4: "orange",
    5: "red",
    6: "skyblue"
}


def plot_bbox(
    image: np.array,
    bboxes: List[List[float]],
    labels: List[float],
    scores: List[float]
) -> None:
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.array(bboxes)
    if not isinstance(bboxes, np.ndarray):
        labels = np.array(labels)
    if not isinstance(bboxes, np.ndarray):
        scores = np.array(scores)
        
    """box is normalized xyxy"""
    image_height, image_width, _ = image.shape

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.imshow(image)
    for bbox, label, score in zip(bboxes, labels, scores):
        x1, y1, x2, y2 = norm_xyxy_to_pixel_xyxy(bbox, image_width, image_height)
        ax.add_patch(Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=1.5,
            edgecolor=color_dict[label],
            facecolor="none"
        ))
        ax.text(
            x1, y1, 
            f"{text_dict[label]} ({score:.2f})", 
            horizontalalignment='left', 
            verticalalignment='bottom', 
            fontsize=12, 
            bbox=dict(facecolor=color_dict[label], alpha=0.3)
        )
    plt.axis("off")
    plt.show()