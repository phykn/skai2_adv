# clear folder
rm -rf /home/work/data
rm -rf /home/work/data_prepared
rm -rf /home/work/src/yolor-main
rm -rf /home/work/src/yolov5-master
rm -rf /home/work/src/yolov7-main
rm -rf /home/work/pretrain

# pip install
pip install -r requirements.txt

# unzip
python src/utils/unzip.py -file "/home/work/sample-notebooks/train.zip" -dst "/home/work/data"
python src/utils/unzip.py -file "/home/work/sample-notebooks/predict.zip" -dst "/home/work/data"
python src/utils/unzip.py -file "/home/work/sample-notebooks/yolor-main.zip" -dst "/home/work/src"
python src/utils/unzip.py -file "/home/work/sample-notebooks/yolov5-master.zip" -dst "/home/work/src"
python src/utils/unzip.py -file "/home/work/sample-notebooks/yolov7-main.zip" -dst "/home/work/src"

# data preparation
python src/data_prepare.py \
--src_csv_path "/home/work/data/train.csv" \
--src_img_folder "/home/work/data/train" \
--dst_csv_path "/home/work/data_prepared/train.csv" \
--dst_img_folder "/home/work/data_prepared/image" \
--img_size 1280 --repeat_l 1 --repeat_m 2 --repeat_s 4

# make label for yolo
python src/yolo_make_label.py \
--src_csv_path "/home/work/data_prepared/train.csv" \
--src_img_folder "/home/work/data_prepared/image" \
--file_column "img_name"

# crop image for classification
python src/crop_image.py \
--src_csv_path "/home/work/data/train.csv" \
--src_img_folder "/home/work/data/train" \
--dst_csv_path "/home/work/data_prepared/train_crop.csv" \
--dst_img_folder "/home/work/data_prepared/image_crop" \
--scale 1.2

# download pretrain weight
python src/download_weight.py \
--dst_folder "/home/work/pretrain"