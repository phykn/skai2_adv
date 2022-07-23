rm -rf inference

python src/yolov5-master/detect.py \
--weights "/home/work/runs/train/yolov5m6_all_epoch_300/weights/last.pt" \
--source "/home/work/data/predict" \
--project "inference" --name "" \
--half --device 0 \
--save-txt --save-conf --save-crop \
--img 1280 --augment


# python src/submission.py \
# --img_folder "/home/work/data/predict" \
# --root "/home/work/inference" \
# --weight "/home/work/runs_clf/classifier_fold_0_denoise/weight/last.pt"

# rm -rf inference/crops
# rm -rf inference/labels