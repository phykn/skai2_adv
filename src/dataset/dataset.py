import random
import numpy as np
import albumentations as A
import torchvision
from typing import List, Tuple
from .open_image import open_image
from .misc import get_unique_indices


class CLF_Dataset:
    def __init__(
        self,
        files: List[str],
        class_ids: List[int],
        background_files: List[str],
        add_data_folder: str,
        background_ratio: float=0.0,
        add_data_ratio: float=0.0,
        img_size: int=224,
        test: bool=False,
    ) -> None:
        self.files = files
        self.class_ids = class_ids
        self.background_files = background_files
        self.background_ratio = background_ratio
        self.add_data_ratio = add_data_ratio
        self.add_data = torchvision.datasets.STL10(
            root=add_data_folder,
            split="unlabeled",
            download=True
        )
        self.test = test

        unique, self.indices = get_unique_indices(class_ids)
        self.unique_class_id = sorted(unique)
        self.setup_transform(img_size=img_size)

    def __len__(
        self
    ) -> int:
        return len(self.files)

    def __getitem__(
        self,
        idx: int
    ) -> dict:
        image, class_id = self.get_data(idx)
        return dict(image=image, class_id=class_id)

    def get_random_idx(
        self
    ) -> int:
        index = np.random.choice(self.unique_class_id) - np.min(self.unique_class_id)
        return random.choice(self.indices[index])

    def setup_transform(
        self,
        img_size: int=224
    ) -> None:
        self.resize = A.Resize(
            height=img_size,
            width=img_size
        )

        self.random_crop = A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=(0.01, 0.10),
            ratio=(0.2, 5.0)
        )

        self.reshape = A.Compose([
            A.HorizontalFlip(
                p=0.5
            ),
            A.RandomRotate90(
                p=0.5
            ),
            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.2,
                    rotate_limit=20,
                    interpolation=1,
                    border_mode=0,
                    value=0,
                    p=1.0
                ),
                A.ElasticTransform(
                    border_mode=0,
                    p=1.0
                )
            ], p=0.5)
        ])

        self.recolor = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.Solarize(
                threshold=[88, 168],
                p=0.5
            )
        ])

        self.noise = A.Compose([
            A.Blur(
                blur_limit=7,
                p=0.5
            )
        ])

        self.normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0
        )

    def transform_image(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray]:
        if not self.test:
            image = self.reshape(image=image)["image"]
            image = self.recolor(image=image)["image"]
            image = self.noise(image=image)["image"]
        image = self.normalize(image=image)["image"]
        image = image.transpose(2, 0, 1)
        return image

    def get_data(
        self,
        idx: int
    ) -> Tuple[np.ndarray, int]:
        if not self.test:
            rand = np.random.rand()
            if  rand < self.background_ratio:
                image = open_image(
                    np.random.choice(self.background_files)
                )
                image = self.random_crop(image=image)["image"]
                class_id = 0
            elif rand < self.background_ratio + self.add_data_ratio:
                image, _ = random.choice(self.add_data)
                image = self.resize(image=np.array(image))["image"]
                class_id = 1
            else:
                idx = self.get_random_idx()
                image = open_image(self.files[idx])
                image = self.resize(image=image)["image"]
                class_id = self.class_ids[idx]
        else:
            image = open_image(self.files[idx])
            image = self.resize(image=image)["image"]
            class_id = self.class_ids[idx]

        return self.transform_image(image), class_id


class Test_Dataset:
    def __init__(
        self,
        files: List[str],
        img_size: int=224
    ) -> None:
        self.files = files
        self.transform = A.Compose([
            A.Resize(
                height=img_size,
                width=img_size
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0
            )
        ])

    def __len__(
        self
    ) -> int:
        return len(self.files)

    def __getitem__(
        self,
        idx: int
    ) -> dict:
        file = self.files[idx]
        image = open_image(file)
        image = self.transform(image=image)["image"]
        image = image.transpose(2, 0, 1)
        return dict(image=image)