# claer data folder
rm -rf /home/work/src/config/split

# split dataset
python src/yolo_split_data.py \
--src_csv_path "/home/work/data_prepared/train.csv" \
--src_img_folder "/home/work/data_prepared/image" \
--dst_txt_folder "/home/work/src/config/split" \
--n_splits 0 --seed 42 \
# --single_object

# yolov5
#python src/yolov5-master/train.py \
#--name yolov5m6_all_epoch_300 \
#--data "/home/work/src/config/split/dataset_00.yaml" \
#--project "/home/work/runs/train" \
#--cfg "/home/work/src/config/yolov5/yolov5m6.yaml" \
#--weights "/home/work/pretrain/yolov5m6.pt" \
#--hyp "/home/work/src/config/yolov5/hyp.yaml" \
#--img 1280 --multi-scale --cos-lr \
#--batch-size 4 --epochs 300 --workers 4 --device 0

# # yolov7
# python src/yolov7-main/train.py \
# --name yolov7x_fold_0 \
# --data "/home/work/src/config/split/dataset_00.yaml" \
# --project "/home/work/runs/train" \
# --cfg "/home/work/src/config/yolov7/yolov7x.yaml" \
# --weights "/home/work/pretrain/yolov7x.pt" \
# --hyp "/home/work/src/config/yolov7/hyp.yaml" \
# --img-size 640 640 --multi-scale \
# --batch-size 4 --epochs 100 --workers 4 --device 0

# # classifier
python src/train_classifier.py \
--name classifier_all_transformer_zero_tgt_300_epoch \
--src_csv_path "/home/work/data_prepared/train_crop.csv" \
--src_img_folder "/home/work/data_prepared/image_crop" \
--bg_img_folder "/home/work/data_prepared/image" \
--n_splits 0 --batch_size 64 --epoch 300 --warmup_steps 5 \
--max_lr 1e-5 --min_lr 1e-6 \
--swa --swa_decay 0.99 --swa_epoch_start 0.8  --swa_annealing_epochs 5