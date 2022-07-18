import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import StratifiedKFold

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning import loggers as pl_loggers

from dataset.dataset import CLF_Dataset
from model.classifier import Classifier
from model.lightning import Lightning


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs_clf", type=str, help="root folder")
    parser.add_argument("--name", default="model", type=str, help="model name")
    parser.add_argument("--src_csv_path", default="data_prepared/train_crop.csv", type=str, help="source csv path")
    parser.add_argument("--src_img_folder", default="data_prepared/image_crop", type=str, help="source image folder")
    parser.add_argument("--bg_img_folder", default="data_prepared/image", type=str, help="background image folder")
    parser.add_argument("--random_state", default=42, type=int, help="random seed")
    
    parser.add_argument("--img_size", default=224, type=int, help="input image size")    
    parser.add_argument("--n_splits", default=10, type=int, help="number of folds")
    parser.add_argument("--fold", default=0, type=int, help="fold index")
    
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="num workers for dataloader")
    parser.add_argument("--pin_memory", default=True, type=str2bool, help="pin memory")
    
    parser.add_argument("--pretrained", default=True, type=str2bool, help="use convnext pretrain weight")
    parser.add_argument("--num_class", default=6, type=int, help="number of class")
    parser.add_argument("--drop_path_rate", default=0.1, type=float, help="drop path rate of convnext")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout")
    parser.add_argument("--d_model", default=256, type=int, help="model dim size")
    parser.add_argument("--n_head", default=8, type=int, help="number of transformer head")
    parser.add_argument("--d_ff", default=1024, type=int, help="feed forward network interlayer dimension")
    parser.add_argument("--num_encoder_layers", default=6, type=int, help="number of transformer encoder layers")
    parser.add_argument("--num_decoder_layers", default=6, type=int, help="number of transformer decoder layers")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="label smoothing")
    
    parser.add_argument("--optimizer", default="adamw", type=str, help="sgd or adamw")
    parser.add_argument("--epoch", default=100, type=int, help="number of epochs")
    parser.add_argument("--warmup_steps", default=5, type=int, help="warmup steps")
    parser.add_argument("--max_lr", default=1e-4, type=float, help="maximum learning rate")
    parser.add_argument("--min_lr", default=1e-5, type=float, help="minimum learning rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight_decay")
    
    parser.add_argument("--ckpt_monitor", default="valid_loss", type=str, help="checkpoint monitor")
    parser.add_argument("--ckpt_mode", default="min", type=str, help="checkpoint mode")
    
    parser.add_argument("--swa", action='store_true', help="enable swa")
    parser.add_argument("--swa_decay", default=0.99, type=float, help="swa decay rate")
    parser.add_argument("--swa_epoch_start", default=0.8, type=float, help="swa start point")
    parser.add_argument("--swa_annealing_epochs", default=5, type=int, help="swa annealing epochs")
    parser.add_argument("--swa_annealing_strategy", default="cos", type=str, help="swa annealing strategy")
    
    parser.add_argument("--gradient_clip_val", default=1.0, type=float, help="gradient clip value")
    parser.add_argument("--accumulate_grad_batches", default=1, type=int, help="accumulate grad batches")
    parser.add_argument("--log_every_n_steps", default=10, type=int, help="log steps")
    parser.add_argument("--precision", default=16, type=int, help="16 of 32")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus")
    
    return parser.parse_args()


def main(args):
    # set folder
    dir_root = args.root
    dir_base = os.path.join(dir_root, args.name)
    dir_ckpt = os.path.join(dir_base, "checkpoint")
    dir_weight = os.path.join(dir_base, "weight")

    os.makedirs(dir_base, exist_ok=True)
    os.makedirs(dir_ckpt, exist_ok=True)
    os.makedirs(dir_weight, exist_ok=True)

    # data
    df = pd.read_csv(args.src_csv_path)
    files = np.array([os.path.join(args.src_img_folder, file) for file in df["file"].values])
    class_ids = df["label"].values - 2
    background_files = sorted(glob(
        os.path.join(args.bg_img_folder, "*_no_obj.jpg")
    ))

    # split data
    if args.n_splits == 0:
        train_loader = DataLoader(
            dataset = CLF_Dataset(
            files=files,
            class_ids=class_ids,
            test=False,
            background_files=background_files
        ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=True,
            drop_last=True
        )

        np.random.seed(args.random_state)
        valid_index = np.random.choice(range(len(files)), size=int(0.2*len(files)), replace=False)

        valid_loader = DataLoader(
            CLF_Dataset(
                files=files[valid_index],
                class_ids=class_ids[valid_index],
                img_size=args.img_size,
                test=True,
                background_files=background_files
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=False,
            drop_last=False
        )

    else:
        skf = StratifiedKFold(
            n_splits=args.n_splits,
            random_state=args.random_state,
            shuffle=True
        )

        for i, (train_index, valid_index) in enumerate(skf.split(files, class_ids)):
            if i == args.fold:
                train_loader = DataLoader(
                    CLF_Dataset(
                        files=files[train_index],
                        class_ids=class_ids[train_index],
                        img_size=args.img_size,
                        test=False,
                        background_files=background_files
                    ),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    shuffle=True,
                    drop_last=True
                )
                valid_loader = DataLoader(
                    CLF_Dataset(
                        files=files[valid_index],
                        class_ids=class_ids[valid_index],
                        img_size=args.img_size,
                        test=True,
                        background_files=background_files
                    ),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    shuffle=False,
                    drop_last=False
                )
                break

    # checkpoint
    checkpoint = ModelCheckpoint(
        dirpath=dir_ckpt,
        monitor=args.ckpt_monitor,
        mode=args.ckpt_mode,
        filename="checkpoint-{epoch:02d}",
        every_n_epochs=1,
        save_last=True,
        save_weights_only=True,
    )

    # lr monitor
    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )

    # logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=dir_base,
        name=args.name
    )

    # callbacks
    callbacks = [checkpoint, lr_monitor]
    if args.swa:
        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
            return args.swa_decay*averaged_model_parameter + (1-args.swa_decay*model_parameter)

        swa = StochasticWeightAveraging(
            swa_epoch_start=args.swa_epoch_start, 
            swa_lrs=args.min_lr, 
            annealing_epochs=args.swa_annealing_epochs,  
            annealing_strategy=args.swa_annealing_strategy,
            avg_fn=avg_fn
        )
        callbacks.append(swa)

    # model
    model = Classifier(
        pretrained=args.pretrained,
        num_class=args.num_class,
        drop_path_rate=args.drop_path_rate,
        dropout=args.dropout,
        d_model=args.d_model,
        n_head=args.n_head,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        label_smoothing=args.label_smoothing
    )

    # lightning
    lightning = Lightning(
        model=model,
        optimizer=args.optimizer,
        first_cycle_steps=args.epoch,
        warmup_steps=args.warmup_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay
    )

    # trainer
    trainer = pl.Trainer(
        max_epochs=args.epoch,
        callbacks = callbacks,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        gpus=args.gpus,
        logger=tb_logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=False
    )

    # train
    trainer.fit(
        model=lightning, 
        train_dataloaders=train_loader, 
        val_dataloaders=valid_loader
    )

    # best model save
    best_model = Lightning.load_from_checkpoint(
        checkpoint_path=checkpoint.best_model_path,
        model=model
    ).model
    torch.save(
        best_model.state_dict(), 
        os.path.join(dir_weight, "best.pt")
    )

    # last model save
    last_model = Lightning.load_from_checkpoint(
        checkpoint_path=checkpoint.last_model_path,
        model=model
    ).model
    torch.save(
        last_model.state_dict(), 
        os.path.join(dir_weight, "last.pt")
    )


if __name__ == "__main__":
    main(get_args())
