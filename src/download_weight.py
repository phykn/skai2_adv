import os
import argparse
import requests


def get_args_parser():
    parser = argparse.ArgumentParser("Download Pretrain Weights", add_help=False)
    parser.add_argument("--dst_folder", default="pretrained", type=str, help="destination pretrain weight folder")
    return parser


def main(args):
    file = requests.get("https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s6.pt", allow_redirects=True)
    with open(os.path.join(args.dst_folder, "yolov5s6.pt"), "wb") as f:
        f.write(file.content)

    file = requests.get("https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m6.pt", allow_redirects=True)
    with open(os.path.join(args.dst_folder, "yolov5m6.pt"), "wb") as f:
        f.write(file.content)

    file = requests.get("https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5l6.pt", allow_redirects=True)
    with open(os.path.join(args.dst_folder, "yolov5l6.pt"), "wb") as f:
        f.write(file.content)

    file = requests.get("https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x6.pt", allow_redirects=True)
    with open(os.path.join(args.dst_folder, "yolov5x6.pt"), "wb") as f:
        f.write(file.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download Pretrain Weights", parents=[get_args_parser()])
    args = parser.parse_args()

    # make dir
    os.makedirs(args.dst_folder, exist_ok=True)

    # main
    main(args)