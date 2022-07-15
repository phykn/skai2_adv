rm -rf inference

python src/yolov5-master/detect.py \
--weights "/home/work/backup/runs_yolov5_0712/train/yolov5s6_class_1_f1_0.0_fold_0/weights/last.pt" \
--source "data/predict" \
--project "inference" --name "" \
--save-txt --save-conf --half --device 0 \
--save-crop \
--img 1280 --augment