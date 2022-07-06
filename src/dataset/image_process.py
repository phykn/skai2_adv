import numpy as np
import albumentations as A
from typing import Tuple, List, Dict
from .bbox import norm_xyxy_to_pixel_xyxy


def delete_objects(
    image: np.ndarray, 
    bboxes: np.ndarray
) -> np.ndarray:
    # image: HxWxC
    # bbox: normalized xyxy

    image_height, image_width, _ = image.shape
    mean = np.mean(image, axis=(0, 1)).astype(int)

    image = image.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = norm_xyxy_to_pixel_xyxy(bbox, image_width, image_height)
        image[y1:y2, x1:x2] = mean

    return image


def extract_objects(
    image: np.ndarray,
    bboxes: np.ndarray,
    class_labels: List[int],
    scale: float=1.2
) -> Tuple[np.ndarray, int]:
    assert scale >= 1.0, "scale must be larger than 0" 
    # image: HxWxC
    # bbox: normalized xyxy

    image_height, image_width, _ = image.shape
    for bbox, class_label in zip(bboxes, class_labels):
        x1, y1, x2, y2 = norm_xyxy_to_pixel_xyxy(bbox, image_width, image_height)
        w = int(np.ceil((x2-x1)*(scale-1)))
        h = int(np.ceil((y2-y1)*(scale-1)))

        x1 = np.maximum(0, x1-w)
        y1 = np.maximum(0, y1-h)
        x2 = np.minimum(x2+w, image_width)
        y2 = np.minimum(y2+h, image_height)
        crop = image[y1:y2, x1:x2]

        yield crop, class_label


def transform_bbox(
    transform: A.Compose,
    image: np.ndarray,
    bboxes: np.ndarray,
    class_labels: np.ndarray
) -> Dict[str, np.ndarray]:
    if class_labels[0] == 0:
        output = A.Compose([transform[0]])(
            image=image
        )
        output["bboxes"] = np.array([[0., 0., 0., 0.]])
        output["class_labels"] = np.array([0])
        return output
    else:
        output = transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        if len(output["bboxes"]) == 0:
            output["bboxes"] = np.array([[0., 0., 0., 0.]])
            output["class_labels"] = np.array([0])
        else:
            output["bboxes"] = np.array(output["bboxes"])
            output["class_labels"] = np.array(output["class_labels"])

        return output


def transform_s():
    return A.Compose(
        [A.RandomResizedCrop(
            height=640,
            width=640,
            scale=(0.05, 0.2),
            ratio=(1.0, 1.0),
            always_apply=True
        )],
        bbox_params=A.BboxParams(
            format='albumentations',
            min_area=128,
            min_visibility=0.2,
            label_fields=["class_labels"]
        )
    )


def transform_m():
    return A.Compose(
        [A.RandomResizedCrop(
            height=640,
            width=640,
            scale=(0.2, 0.4),
            ratio=(1.0, 1.0),
            always_apply=True
        )],
        bbox_params=A.BboxParams(
            format='albumentations',
            min_area=128,
            min_visibility=0.2,
            label_fields=["class_labels"]
        )
    )


def transform_l():
    return A.Compose(
        [A.RandomResizedCrop(
            height=640,
            width=640,
            scale=(0.4, 0.8),
            ratio=(1.0, 1.0),
            always_apply=True
        )],
        bbox_params=A.BboxParams(
            format='albumentations',
            min_area=128,
            min_visibility=0.2,
            label_fields=["class_labels"]
        )
    )