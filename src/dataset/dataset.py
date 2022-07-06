import random
import albumentations as A
from typing import List
from .open_image import open_image
from ..utils.misc import get_unique_indices


class CLF_Dataset:
    def __init__(
        self,
        files: List[str],
        labels: List[int],
        img_size: int=224,
        random_select: bool=False,
        random_state: int=42
    ) -> None:
        self.files = files
        self.labels = labels
        self.random_select = random_select
        self.random_state = random_state

        random.seed(random_state)
        unique, self.indices = get_unique_indices(labels)
        self.num_class = len(unique)

        self.b_transform = A.Compose([
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
                    )
                ])

        self.n_transform = A.Compose([
            A.Blur(
                blur_limit=7,
                p=0.5
            ),
            A.GaussNoise(
                p=0.5
            ),
            A.CoarseDropout(
                max_height=112,
                max_width=112,
                max_holes=4,
                min_height=32,
                min_width=32,
                min_holes=1,
                p=1.0
            )
        ])

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
        
        target_image = self.b_transform(image=image)["image"]
        input_image = self.n_transform(image=target_image)["image"]
        
        target_image = self.normalize(image=target_image)["image"].transpose(2, 0, 1)
        input_image = self.normalize(image=input_image)["image"].transpose(2, 0, 1)

        return dict(
            input_image=input_image,
            target_image=target_image,
            label=label
        )

    def _get_random_idx(
        self
    ) -> int:
        return random.choice(
            self.indices[random.randint(0, self.num_class-1)]
        )