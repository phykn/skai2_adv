import numpy as np
from typing import List, Optional


def non_max_suppression_index(
    bboxes: np.ndarray, 
    overlapThresh: float=0.8, 
    probs: Optional[List[float]]=None,
    eps: float=1e-7
) -> List[int]:
    # Source: https://github.com/bruceyang2012/nms_python/blob/master/nms.py

    pick = []
    if len(bboxes) == 0:
        return pick

    x1s = bboxes[:, 0]
    y1s = bboxes[:, 1]
    x2s = bboxes[:, 2]
    y2s = bboxes[:, 3]

    area = (x2s - x1s) * (y2s - y1s)
    idxs = y2s

    if probs is not None:
        idxs = probs

    idxs = np.argsort(idxs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1s[i], x1s[idxs[:last]])
        yy1 = np.maximum(y1s[i], y1s[idxs[:last]])
        xx2 = np.minimum(x2s[i], x2s[idxs[:last]])
        yy2 = np.minimum(y2s[i], y2s[idxs[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w*h) / (area[idxs[:last]] + eps)

        idxs = np.delete(
            idxs,
            np.concatenate(
                ([last], np.where(overlap > overlapThresh)[0])
            )
        )

    return pick