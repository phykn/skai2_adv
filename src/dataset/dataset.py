import random
import numpy as np
import albumentations as A
from typing import List, Tuple
from .open_image import open_image
from .misc import get_unique_indices


class CLF_Dataset:
    def __init__(
        self,
        files: List[str],
        labels: List[int],
        img_size: int=224,
        random_select: bool=False,
        test: bool=False
    ) -> None:
        self.files = files
        self.labels = labels
        self.random_select = random_select
        self.test = test

        unique, self.indices = get_unique_indices(labels)
        self.num_class = len(unique)

        r_transform = [
            A.Resize(
                height=img_size, 
                width=img_size
            )
        ]
        self.r_transform = A.Compose(r_transform)

        b_transform = [
            A.Resize(
                height=img_size, 
                width=img_size
            ),
            A.HorizontalFlip(
                p=0.5
            ),
            A.RandomRotate90(
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=20,
                interpolation=1,
                border_mode=0,
                value=0,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.ElasticTransform(
                border_mode=0, 
                p=0.5
            )
        ]
        self.b_transform = A.Compose(b_transform)

        n_transform = [
            A.Blur(
                blur_limit=7, 
                p=0.5
            ),
            A.GaussNoise(
                p=0.5
            ),
            A.OneOf(
                [
                    A.CoarseDropout(
                        max_height=112,
                        max_width=112,
                        max_holes=4,
                        min_height=32,
                        min_width=32,
                        min_holes=1,
                        p=1.0
                    ),
                    A.ToGray(
                        p=1.0
                    ),
                    A.Solarize(
                        threshold=[88, 168],
                        p=1.0
                    )
                ],
                p=0.5
            )
        ]
        self.n_transform = A.Compose(n_transform)

        self.normalize = A.Compose([
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
        if self.random_select:
            idx = self._get_random_idx()

        file = self.files[idx]
        label = self.labels[idx]

        image = open_image(file)
        input_image, target_image = self.transform_image(image)

        return dict(
            input_image=input_image,
            target_image=target_image,
            label=label
        )

    def _get_random_idx(
        self
    ) -> int:
        l = np.random.choice(range(0, self.num_class), size=1, replace=False)[0]
        return random.choice(self.indices[l])

    def transform_image(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray]:
        if self.test:
            target_image = self.r_transform(image=image)["image"]
            input_image = target_image.copy()
        else:
            target_image = self.b_transform(image=image)["image"]
            input_image = self.n_transform(image=target_image)["image"]

        target_image = self.normalize(image=target_image)["image"].transpose(2, 0, 1)
        input_image = self.normalize(image=input_image)["image"].transpose(2, 0, 1)

        return input_image, target_image


class Twin_CLF_Dataset(CLF_Dataset):
    def __init__(
        self,
        files: List[str],
        labels: List[int],
        img_size: int=224,
        test: bool=False
    ) -> None:
        self.files = files
        super().__init__(
            files=files, 
            labels=labels, 
            img_size=img_size, 
            test=test
        )

    def __len__(
        self
    ) -> int:
        return 2*len(self.files)

    def _get_random_twin_idx(
        self
    ) -> Tuple[int]:
        same_label = random.choice([False, True])
        if same_label:
            l = np.random.choice(range(0, self.num_class), size=1, replace=False)[0]
            return random.choice(self.indices[l]), random.choice(self.indices[l])
        else:
            l0, l1 = np.random.choice(range(0, self.num_class), size=2, replace=False)
            return random.choice(self.indices[l0]), random.choice(self.indices[l1])

    def __getitem__(
        self,
        idx: int
    ) -> dict:
        idx_0, idx_1 = self._get_random_twin_idx()

        file_0 = self.files[idx_0]
        label_0 = self.labels[idx_0]
        input_image_0, target_image_0 = self.transform_image(open_image(file_0))

        file_1 = self.files[idx_1]
        label_1 = self.labels[idx_1]
        input_image_1, target_image_1 = self.transform_image(open_image(file_1))

        return dict(
            data_0=dict(
                input_image=input_image_0,
                target_image=target_image_0,
                label=label_0
            ),
            data_1=dict(
                input_image=input_image_1,
                target_image=target_image_1,
                label=label_1
            )
        )
    
    
class Test_Dataset:
    def __init__(
        self,
        files: List[str],
        img_size: int=224,
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
        image = self.transform(image=image)["image"].transpose(2, 0, 1)

        return dict(
            input_image=image
        )