# claer data folder
rm -rf /home/work/src/config/split
# rm -rf /home/work/runs

# split dataset
python src/yolo_split_data.py \
--src_csv_path "/home/work/data_prepared/train.csv" \
--src_img_folder "/home/work/data_prepared/image" \
--dst_txt_folder "/home/work/src/config/split" \
--n_splits 10 --single_object --seed 42

### yolov5 ###

# # Fold: 0
# python src/yolov5-master/train.py \
# --name yolov5s6_class_1_f1_0.0_fold_0 \
# --data "/home/work/src/config/split/dataset_00.yaml" \
# --project "/home/work/runs/train" \
# --cfg "/home/work/src/config/yolov5/yolov5s6_class_1.yaml" \
# --weights "/home/work/pretrained/yolov5s6.pt" \
# --hyp "/home/work/src/config/yolov5/f1_gamma/hyp_0.0.yaml" \
# --img 1280 --multi-scale --label-smoothing 0.1 --cos-lr --single-cls \
# --batch-size 2 --epochs 50 --workers 4 --device 0

# # Fold: 0
# python src/yolov5-master/train.py \
# --name yolov5s6_class_1_f1_1.0_fold_0 \
# --data "/home/work/src/config/split/dataset_00.yaml" \
# --project "/home/work/runs/train" \
# --cfg "/home/work/src/config/yolov5/yolov5s6_class_1.yaml" \
# --weights "/home/work/pretrained/yolov5s6.pt" \
# --hyp "/home/work/src/config/yolov5/f1_gamma/hyp_1.0.yaml" \
# --img 1280 --multi-scale --cos-lr \
# --batch-size 4 --epochs 50 --workers 6 --device 0

# # # Fold: 0
# python src/yolov5-master/train.py \
# --name yolov5s6_class_1_f1_0.5_fold_0 \
# --data "/home/work/src/config/split/dataset_00.yaml" \
# --project "/home/work/runs/train" \
# --cfg "/home/work/src/config/yolov5/yolov5s6_class_1.yaml" \
# --weights "/home/work/pretrained/yolov5s6.pt" \
# --hyp "/home/work/src/config/yolov5/f1_gamma/hyp_0.5.yaml" \
# --img 1280 --multi-scale --cos-lr \
# --batch-size 4 --epochs 50 --workers 6 --device 0

# # Fold: 0
# python src/yolov5-master/train.py \
# --name yolov5s6_class_1_f1_1.5_fold_0 \
# --data "/home/work/src/config/split/dataset_00.yaml" \
# --project "/home/work/runs/train" \
# --cfg "/home/work/src/config/yolov5/yolov5s6_class_1.yaml" \
# --weights "/home/work/pretrained/yolov5s6.pt" \
# --hyp "/home/work/src/config/yolov5/f1_gamma/hyp_1.5.yaml" \
# --img 1280 --multi-scale --label-smoothing 0.1 --cos-lr --single-cls \
# --batch-size 4 --epochs 50

# # Fold: 0
# python src/yolov5-master/train.py \
# --name yolov5s6_class_1_f1_2.0_fold_0 \
# --data "/home/work/src/config/split/dataset_00.yaml" \
# --project "/home/work/runs/train" \
# --cfg "/home/work/src/config/yolov5/yolov5s6_class_1.yaml" \
# --weights "/home/work/pretrained/yolov5s6.pt" \
# --hyp "/home/work/src/config/yolov5/f1_gamma/hyp_2.0.yaml" \
# --img 1280 --multi-scale --cos-lr \
# --batch-size 4 --epochs 50 --workers 6 --device 0


### yolor ###

# # Fold: 0
# python src/yolor-main/train.py \
# --name yolor_p6_class_1_fold_0 \
# --data "/home/work/src/config/split/dataset_00.yaml" \
# --cfg "/home/work/src/config/yolor/yolor_p6_class_1.cfg" \
# --weights "/home/work/pretrained/yolor_p6.pt" \
# --hyp "/home/work/src/config/yolor/hyp.yaml" \
# --img 640 640 --multi-scale \
# --batch-size 4 --epochs 50 --workers 4 --device 0

### classifier ###
python src/train_classifier.py \
--name classifier_00 \
--src_csv_path "/home/work/data_prepared/train_crop.csv" \
--src_img_folder "/home/work/data_prepared/image_crop" \
--label_smoothing 0.1 --warmup_steps 0 --epoch 1 --swa --swa_decay 0.99