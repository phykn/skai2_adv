import numpy as np
import pandas as pd


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