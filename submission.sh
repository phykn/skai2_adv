rm -rf inference

python src/yolov5-master/detect.py \
--weights \
"/home/work/runs/train/yolov5m_epoch_100/weights/last.pt" \
"/home/work/runs/train/yolov5s6_epoch_100/weights/last.pt" \
"/home/work/runs/train/yolov5m6_epoch_100/weights/last.pt" \
--source "/home/work/data/predict" \
--project "inference" --name "" \
--img 2560 --max-det 300 --conf-thres 0.3 --iou-thres 0.1 --augment \
--nosave --save-txt --save-conf --save-crop --half --device 0

python src/submission.py \
--img_folder "/home/work/data/predict" \
--inference_folder "/home/work/inference" \
--output "submission.csv" \
--clf_weight "/home/work/runs_clf/classifier_all_bg_0.2_add_0.2_epoch_200/weight/last.pt" \
--use_clf true --clf_ratio 0.1 --conf_thres 0.0 --overlapThresh 1.0 --small_object_limit 0.0075

rm -rf inference/crops
rm -rf inference/labels