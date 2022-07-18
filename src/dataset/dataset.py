import random
import numpy as np
import albumentations as A
from typing import List, Tuple, Optional
from .open_image import open_image
from .misc import get_unique_indices


class CLF_Dataset:
    def __init__(
        self,
        files: List[str],
        class_ids: List[int],
        img_size: int=224,
        test: bool=False,
        background_files: Optional[List[str]]=None
    ) -> None:
        self.files = files
        self.class_ids = class_ids
        self.test = test
        self.background_files = background_files

        unique, self.indices = get_unique_indices(class_ids)
        self.num_class = len(unique)
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
        class_id = np.random.choice(range(0, self.num_class))
        return random.choice(self.indices[class_id])

    def setup_transform(
        self,
        img_size: int=224
    ) -> None:
        self.resize = A.Resize(
            height=img_size,
            width=img_size
        )

        self.random_crop = A.RandomSizedCrop(
            min_max_height=(img_size//2, img_size*4),
            height=img_size,
            width=img_size
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
            ], p=1.0)
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
            ),
            A.GaussNoise(
                p=0.5
            ),
            A.CoarseDropout(
                max_height=56,
                max_width=56,
                max_holes=8,
                min_height=28,
                min_width=28,
                min_holes=4,
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
            idx = self.get_random_idx()            
            if (self.background_files is not None) and (np.random.rand() < 1/(self.num_class+1)):
                image = open_image(
                    np.random.choice(self.background_files)
                )
                image = self.random_crop(image=image)["image"]
                return self.transform_image(image), self.num_class

        image = open_image(self.files[idx])
        image = self.resize(image=image)["image"]
        return self.transform_image(image), self.class_ids[idx]


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
        image = self.transform(image=image)["image"].transpose(2, 0, 1)
        return dict(image=image)
