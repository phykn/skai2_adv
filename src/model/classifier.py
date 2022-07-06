import torch
import torch.nn as nn
from typing import Dict
from timm.loss import LabelSmoothingCrossEntropy
from .convnext import convnext_tiny
from .layers import ImageGenerator


class Classifier(nn.Module):
    def __init__(
        self,
        num_class: int=5,
        pretrained: bool=True,
        drop_path_rate: float=0.1,
        label_smoothing: float=0.1
    ) -> None:
        super().__init__()
        self.convnext = convnext_tiny(pretrained=True, drop_path_rate=0.1)
        self.convnext.head = nn.Linear(self.convnext.head.in_features, num_class, bias=True)
        self.ig = ImageGenerator([768, 384, 192, 96, 64, 3])
        
        self.ce_loss = LabelSmoothingCrossEntropy(label_smoothing)
        self.l1_loss = nn.L1Loss()
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        out = self.convnext(x)
        logit = out["y"]
        image = self.ig(out["xs"])
        return dict(logit=out["y"], image=image)

    def loss(
        self,
        data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out = self(data["input_image"])
        ce_loss = self.ce_loss(out["logit"], data["label"])
        l1_loss = self.l1_loss(out["image"], data["target_image"])
        return dict(
            clf_loss=ce_loss, 
            img_loss=l1_loss, 
            loss=ce_loss+l1_loss
        )