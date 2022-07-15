import os
import argparse
import requests


def get_args_parser():
    parser = argparse.ArgumentParser("Download Pretrain Weights", add_help=False)
    parser.add_argument("--dst_folder", default="pretrain", type=str, help="destination pretrain weight folder")
    return parser


def download_yolo_weight(
    dst_folder: str,
    name: str
) -> None:
    file = requests.get(f"https://github.com/ultralytics/yolov5/releases/download/v6.1/{name}", allow_redirects=True)
    with open(os.path.join(args.dst_folder, name), "wb") as f:
        f.write(file.content)
    print(f"Download: {name}")
    

def main(args):
    download_yolo_weight(args.dst_folder, "yolov5s6.pt")
    download_yolo_weight(args.dst_folder, "yolov5m6.pt")
    download_yolo_weight(args.dst_folder, "yolov5l6.pt")
    download_yolo_weight(args.dst_folder, "yolov5x6.pt")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download Pretrain Weights", parents=[get_args_parser()])
    args = parser.parse_args()

    # make dir
    os.makedirs(args.dst_folder, exist_ok=True)

    # main
    main(args)