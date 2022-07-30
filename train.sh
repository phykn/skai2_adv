# claer data folder
rm -rf /home/work/src/config/split

# split dataset
python src/yolo_split_data.py \
--src_csv_path "/home/work/data_prepared/train.csv" \
--src_img_folder "/home/work/data_prepared/image" \
--dst_txt_folder "/home/work/src/config/split" \
--n_splits 0 --seed 42

# # yolov5
# python src/yolov5-master/train.py \
# --name yolov5m_epoch_100 \
# --data "/home/work/src/config/split/dataset_00.yaml" \
# --project "/home/work/runs/train" \
# --cfg "/home/work/src/config/yolov5/yolov5m.yaml" \
# --weights "/home/work/pretrain/yolov5m.pt" \
# --hyp "/home/work/src/config/yolov5/hyp.yaml" \
# --img 640 --multi-scale --cos-lr \
# --batch-size 4 --epochs 100 --workers 4 --device 0

# python src/yolov5-master/train.py \
# --name yolov5l_epoch_100 \
# --data "/home/work/src/config/split/dataset_00.yaml" \
# --project "/home/work/runs/train" \
# --cfg "/home/work/src/config/yolov5/yolov5l.yaml" \
# --weights "/home/work/pretrain/yolov5l.pt" \
# --hyp "/home/work/src/config/yolov5/hyp.yaml" \
# --img 640 --multi-scale --cos-lr \
# --batch-size 4 --epochs 100 --workers 4 --device 0

# python src/yolov5-master/train.py \
# --name yolov5s6_epoch_100 \
# --data "/home/work/src/config/split/dataset_00.yaml" \
# --project "/home/work/runs/train" \
# --cfg "/home/work/src/config/yolov5/yolov5s6.yaml" \
# --weights "/home/work/pretrain/yolov5s6.pt" \
# --hyp "/home/work/src/config/yolov5/hyp.yaml" \
# --img 1280 --multi-scale --cos-lr \
# --batch-size 4 --epochs 100 --workers 4 --device 0

# python src/yolov5-master/train.py \
# --name yolov5m6_epoch_100 \
# --data "/home/work/src/config/split/dataset_00.yaml" \
# --project "/home/work/runs/train" \
# --cfg "/home/work/src/config/yolov5/yolov5m6.yaml" \
# --weights "/home/work/pretrain/yolov5m6.pt" \
# --hyp "/home/work/src/config/yolov5/hyp.yaml" \
# --img 1280 --multi-scale --cos-lr \
# --batch-size 4 --epochs 100 --workers 4 --device 0

# classifier
python src/train_classifier.py \
--name classifier_all_bg_0.2_add_0.2_epoch_200 \
--src_csv_path "/home/work/data_prepared/train_crop.csv" \
--src_img_folder "/home/work/data_prepared/image_crop" \
--bg_img_folder "/home/work/data_prepared/image" \
--add_data_folder "/home/work/data_prepared/additional" \
--num_encoder_layers 6 --num_decoder_layers 6 --num_class 7 \
--bg_ratio 0.2 --add_ratio 0.2 --img_size 224 --n_splits 0 --fold 0 \
--batch_size 64 --epoch 200 --first_cycle_steps 100 --cycle_mult 1.0 --warmup_steps 0 --gamma 0.1 --max_lr 1e-4 --min_lr 1e-6
# --swa --swa_decay 0.9 --swa_epoch_start 0.8  --swa_annealing_epochs 0