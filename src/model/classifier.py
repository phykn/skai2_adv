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
        return dict(logit=logit, image=image)

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
    
    def twin(
        self,
        data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out = self.convnext(data["input_image"])
        logit = out["y"]
        image = self.ig(out["xs"])
        embed = out["embed"]
        
        ce_loss = self.ce_loss(logit, data["label"])
        l1_loss = self.l1_loss(image, data["target_image"])
        
        return dict(
            embed=embed,
            clf_loss=ce_loss, 
            img_loss=l1_loss, 
            loss=ce_loss+l1_loss
        )
    
    
class Twin_Classifier(nn.Module):
    def __init__(
        self,
        num_class: int=5,
        pretrained: bool=True,
        drop_path_rate: float=0.1,
        label_smoothing: float=0.1
    ) -> None:
        super().__init__()
        self.clf = Classifier(
            num_class=num_class,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            label_smoothing=label_smoothing
        )
        self.pair = nn.Linear(3*self.clf.convnext.head.in_features, 2, bias=True)
        self.ce_loss = LabelSmoothingCrossEntropy(label_smoothing)
        
    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> torch.Tensor:        
        return dict(
            y0=self.clf(x0),
            y1=self.clf(x1)
        )
    
    def predict(
        self,
        data: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return self(
            data["data_0"]["input_image"],
            data["data_1"]["input_image"]
        )

    def loss(
        self,
        data: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        data_0 = data["data_0"]
        data_1 = data["data_1"]
        
        twin_0 = self.clf.twin(data_0)
        twin_1 = self.clf.twin(data_1)
        
        p_pair = self.pair(
            torch.cat([
                twin_0["embed"],
                twin_1["embed"],
                torch.abs(twin_0["embed"]-twin_1["embed"])
            ], axis=1)
        )
        t_pair = 1*(data["data_0"]["label"]==data["data_1"]["label"])
        
        pair_loss = self.ce_loss(p_pair, t_pair)
        
        return dict(
            clf_loss_0=twin_0["clf_loss"],
            img_loss_0=twin_0["img_loss"],
            clf_loss_1=twin_1["clf_loss"],
            img_loss_1=twin_1["img_loss"],
            pair_loss=pair_loss,
            loss=twin_0["loss"]+twin_1["loss"]+pair_loss
        )