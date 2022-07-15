# claer data folder
rm -rf /home/work/src/config/split

# # split dataset
# python src/yolo_split_data.py \
# --src_csv_path "/home/work/data_prepared/train.csv" \
# --src_img_folder "/home/work/data_prepared/image" \
# --dst_txt_folder "/home/work/src/config/split" \
# --n_splits 10 --single_object --seed 42

### yolov5 ###
# python src/yolov5-master/train.py \
# --name yolov5m6_class_1_f1_0.0_fold_0 \
# --data "/home/work/src/config/split/dataset_00.yaml" \
# --project "/home/work/runs/train" \
# --cfg "/home/work/src/config/yolov5/yolov5m6_class_1.yaml" \
# --weights "/home/work/pretrained/yolov5m6.pt" \
# --hyp "/home/work/src/config/yolov5/f1_gamma/hyp_0.0.yaml" \
# --img 1280 --multi-scale --label-smoothing 0.1 --cos-lr --single-cls \
# --batch-size 2 --epochs 50 --workers 4 --device 0

### classifier ###
python src/train_classifier.py \
--name classifier_fold_0_denoise \
--src_csv_path "/home/work/data_prepared/train_crop.csv" \
--src_img_folder "/home/work/data_prepared/image_crop" \
--batch_size 64 --epoch 50 --warmup_steps 5 \
--max_lr 1e-5 --min_lr 1e-6 \
--swa --swa_decay 0.99 --swa_epoch_start 0.8  --swa_annealing_epochs 0