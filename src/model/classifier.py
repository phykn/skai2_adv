import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from timm.loss import LabelSmoothingCrossEntropy
from .convnext import convnext_tiny
from .layers import ListLayerNorm, ImageGenerator, PositionEmbeddingSine, ChangeKernel, GlobalAveragePooling2D, AttentionEncodeLayer


class Classifier(nn.Module):
    def __init__(
        self,
        num_class: int=5,
        pretrained: bool=True,
        drop_path_rate: float=0.1,
        num_encode_layer: int=3,
        hid_dim: int=256,
        num_head: int=8,
        dropout: float=0.1,
        label_smoothing: float=0.1
    ) -> None:
        super().__init__()
        self.kernels = [768, 384, 192, 96, 64, 3]

        self.convnext = convnext_tiny(pretrained=True, drop_path_rate=0.1)
        self.norm = ListLayerNorm(self.kernels[:4][::-1])

        self.img_gen = ImageGenerator(self.kernels)

        self.cng_kernel = ChangeKernel(self.kernels[0], hid_dim)
        self.pos_encode = PositionEmbeddingSine(num_pos_feats=hid_dim//2, normalize=True)
        self.att_encode = nn.Sequential(
            *[AttentionEncodeLayer(d_model=hid_dim, n_head=num_head, d_ff=hid_dim*4, dropout=dropout) for _ in range(num_encode_layer)]
        )

        self.pooling = GlobalAveragePooling2D(hid_dim)
        self.head = nn.Linear(hid_dim, num_class)

        self.ce_loss = LabelSmoothingCrossEntropy(label_smoothing)
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # convnext
        xs = self.convnext(x)
        xs = self.norm(xs)

        # add noise
        if self.training:
            xs[-1] = xs[-1] + torch.randn_like(xs[-1])

        # generate image
        image = self.img_gen(xs)

        # attention
        x = xs[-1]
        x = self.cng_kernel(x)
        p = self.pos_encode(x)
        for encoder in self.att_encode:
            x, score = encoder(x, p)

        # pooling
        embed = self.pooling(x)

        # logit
        logit = self.head(embed)

        return dict(
            logit=logit,
            image=image,
            score=score,
            embed=embed
        )
   
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
        out = self(data["input_image"])
        ce_loss = self.ce_loss(out["logit"], data["label"])
        l1_loss = self.l1_loss(out["image"], data["target_image"])
        accuracy = np.mean(
            torch.argmax(out["logit"], axis=1).detach().cpu().numpy()==data["label"].cpu().numpy()
        )

        return dict(
            clf_loss=ce_loss,
            img_loss=l1_loss,
            loss=(ce_loss+l1_loss)/2,
            embed=out["embed"],
            accuracy=torch.tensor(accuracy)
        )


class Twin_Classifier(nn.Module):
    def __init__(
        self,
        num_class: int=5,
        pretrained: bool=True,
        drop_path_rate: float=0.1,
        num_encode_layer: int=3,
        hid_dim: int=256,
        num_head: int=8,
        dropout: float=0.1,
        label_smoothing: float=0.1
    ) -> None:
        super().__init__()
        self.clf = Classifier(
            num_class=num_class,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            num_encode_layer=num_encode_layer,
            hid_dim=hid_dim,
            num_head=num_head,
            dropout=dropout,
            label_smoothing=label_smoothing
        )
        self.pair = nn.Linear(3*hid_dim, 2, bias=True)
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

        pair_acc = np.mean(
            torch.argmax(p_pair, axis=1).detach().cpu().numpy()==t_pair.cpu().numpy()
        )
        return dict(
            clf_loss_0=twin_0["clf_loss"],
            img_loss_0=twin_0["img_loss"],
            accuracy_0=twin_0["accuracy"],
            clf_loss_1=twin_1["clf_loss"],
            img_loss_1=twin_1["img_loss"],
            accuracy_1=twin_1["accuracy"],
            pair_loss=pair_loss,
            pair_acc=torch.tensor(pair_acc),
            loss=(twin_0["loss"]+twin_1["loss"]+pair_loss)/3
        )