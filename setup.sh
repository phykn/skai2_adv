# clear folder
rm -rf /home/work/data
rm -rf /home/work/data_prepared
rm -rf /home/work/src/yolor-main
rm -rf /home/work/src/yolov5-master

# pip install
pip install -r requirements.txt

# unzip
python src/utils/unzip.py -file "/home/work/sample-notebooks/train.zip" -dst "/home/work/data"
python src/utils/unzip.py -file "/home/work/sample-notebooks/predict.zip" -dst "/home/work/data"
python src/utils/unzip.py -file "/home/work/sample-notebooks/yolor-main.zip" -dst "/home/work/src"
python src/utils/unzip.py -file "/home/work/sample-notebooks/yolov5-master.zip" -dst "/home/work/src"

# data preparation
python src/data_prepare.py \
--src_csv_path "/home/work/data/train.csv" \
--src_img_folder "/home/work/data/train" \
--dst_csv_path "/home/work/data_prepared/train.csv" \
--dst_img_folder "/home/work/data_prepared/image" \
--repeat_l 2 --repeat_m 4 --repeat_s 8

# make label for yolo
python src/yolo_make_label.py \
--src_csv_path "/home/work/data_prepared/train.csv" \
--src_img_folder "/home/work/data_prepared/image" \
--file_column "img_name" \
--single_object