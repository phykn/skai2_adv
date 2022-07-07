import pytorch_lightning as pl
from ..utils.scheduler import CosineAnnealingWarmupRestarts


class Lightning(pl.LightningModule):
    def __init__(
        self,
        model: Module,
        optimizer: str="sgd",
        betas: List[float]=(0.5, 0.999),
        weight_decay: float=0,
        first_cycle_steps: int=100,
        cycle_mult: float=1.0,
        max_lr: float=0.01,
        min_lr: float=0.001,
        warmup_steps: int=5,
        gamma: float=1.0
        
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.betas = betas
        self.weight_decay = weight_decay     
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma   

    def forward(self, data: dict):
        return self.model.predict(data)

    def training_step(self, batch, batch_idx):
        loss = self.model.loss(batch)
        for key in loss.keys():
            self.log(
                name=f"train_{key}", 
                value=loss[key]
            )
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        return self.model.loss(batch)

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        for key in keys:
            self.log(
                name=f"valid_{key}", 
                value=torch.stack([output[key] for output in outputs]).mean()
            )

    def configure_optimizers(self):
        if self.optimizer=="sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.min_lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True
            )
        elif self.optimizer=="adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.min_lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"{self.optimizer} optimizer is not supported.")

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, 
            first_cycle_steps=self.first_cycle_steps,
            cycle_mult=self.cycle_mult,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_steps=self.warmup_steps,
            gamma=self.gamma
        )
        return [optimizer], [{"scheduler": scheduler}]