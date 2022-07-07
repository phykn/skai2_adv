# claer data folder
rm -rf /home/work/src/config/split
rm -rf /home/work/runs

# split dataset
python src/yolo_split_data.py \
--src_csv_path "/home/work/data_prepared/train.csv" \
--src_img_folder "/home/work/data_prepared/image" \
--dst_txt_folder "/home/work/src/config/split" \
--n_splits 10 --single_object --seed 42

### yolov5 ###

# # Fold: 0
# python src/yolov5-master/train.py \
# --name yolov5x_class_1_fold_0 \
# --data "/home/work/src/config/split/dataset_00.yaml" \
# --project "/home/work/runs/train" \
# --cfg "/home/work/src/config/yolov5/yolov5x_class_1.yaml" \
# --weights "/home/work/pretrained/yolov5x.pt" \
# --hyp "/home/work/src/config/yolov5/hyp.yaml" \
# --img 640 --multi-scale --label-smoothing 0.1 \
# --batch-size 4 --epochs 50 --workers 4 --device 0

# Fold: 0
python src/yolov5-master/train.py \
--name yolov5s6_class_1_fold_0 \
--data "/home/work/src/config/split/dataset_00.yaml" \
--project "/home/work/runs/train" \
--cfg "/home/work/src/config/yolov5/yolov5s6_class_1.yaml" \
--weights "/home/work/pretrained/yolov5s6.pt" \
--hyp "/home/work/src/config/yolov5/hyp.yaml" \
--img 640 --multi-scale --label-smoothing 0.1 \
--batch-size 4 --epochs 100 --workers 4 --device 0

# Fold: 0
python src/yolov5-master/train.py \
--name yolov5m6_class_1_fold_0 \
--data "/home/work/src/config/split/dataset_00.yaml" \
--project "/home/work/runs/train" \
--cfg "/home/work/src/config/yolov5/yolov5m6_class_1.yaml" \
--weights "/home/work/pretrained/yolov5m6.pt" \
--hyp "/home/work/src/config/yolov5/hyp.yaml" \
--img 640 --multi-scale --label-smoothing 0.1 \
--batch-size 4 --epochs 100 --workers 4 --device 0

# Fold: 0
python src/yolov5-master/train.py \
--name yolov5l6_class_1_fold_0 \
--data "/home/work/src/config/split/dataset_00.yaml" \
--project "/home/work/runs/train" \
--cfg "/home/work/src/config/yolov5/yolov5l6_class_1.yaml" \
--weights "/home/work/pretrained/yolov5l6.pt" \
--hyp "/home/work/src/config/yolov5/hyp.yaml" \
--img 640 --multi-scale --label-smoothing 0.1 \
--batch-size 4 --epochs 100 --workers 4 --device 0

# Fold: 0
python src/yolov5-master/train.py \
--name yolov5x6_class_1_fold_0 \
--data "/home/work/src/config/split/dataset_00.yaml" \
--project "/home/work/runs/train" \
--cfg "/home/work/src/config/yolov5/yolov5x6_class_1.yaml" \
--weights "/home/work/pretrained/yolov5x6.pt" \
--hyp "/home/work/src/config/yolov5/hyp.yaml" \
--img 640 --multi-scale --label-smoothing 0.1 \
--batch-size 4 --epochs 100 --workers 4 --device 0


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