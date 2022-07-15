rm -rf inference

python src/yolov5-master/detect.py \
--weights "/home/work/backup/runs_yolov5_0712/train/yolov5s6_class_1_f1_0.0_fold_0/weights/last.pt" \
--source "/home/work/data/predict" \
--project "inference" --name "" \
--save-txt --save-conf --half --device 0 \
--nosave --save-crop \
--img 1280 --augment

python src/submission.py \
--img_folder "/home/work/data/predict" \
--root "/home/work/inference" \
--weight "/home/work/runs_clf/classifier_fold_0_denoise/weight/last.pt"

rm -rf inference/crops
rm -rf inference/labels