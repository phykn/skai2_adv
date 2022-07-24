rm -rf inference

python src/yolov5-master/detect.py \
--weights "/home/work/runs/train/yolov5m6_all_epoch_300/weights/last.pt" \
--source "/home/work/data/predict" \
--project "inference" --name "" \
--conf-thres 0.25 --iou-thres 0.45 --half --device 0 \
--nosave --save-txt --save-conf --save-crop \
--img 2560 --augment


python src/submission.py \
--img_folder "/home/work/data/predict" \
--inference_folder "/home/work/inference" \
--output "submission.csv" \
--use_clf false --img_size 224 --batch_size 32 --num_workers 4 --cuda true \
--clf_weight "/home/work/runs_clf/classifier_all_transformer_zero_tgt_300_epoch/weight/last.pt"

rm -rf inference/crops
rm -rf inference/labels