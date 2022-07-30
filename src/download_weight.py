import os
import argparse
import requests


def get_args():
    parser = argparse.ArgumentParser("Download Pretrain Weights", add_help=False)
    parser.add_argument("--dst_folder", default="pretrain", type=str, help="destination pretrain weight folder")
    return parser.parse_args()


def download_yolov5_weight(
    dst_folder: str,
    name: str
) -> None:
    file = requests.get(f"https://github.com/ultralytics/yolov5/releases/download/v6.1/{name}", allow_redirects=True)
    with open(os.path.join(args.dst_folder, name), "wb") as f:
        f.write(file.content)
    print(f"Download: {name}")

    
def download_yolov7_weight(
    dst_folder: str,
    name: str
) -> None:
    file = requests.get(f"https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{name}", allow_redirects=True)
    with open(os.path.join(args.dst_folder, name), "wb") as f:
        f.write(file.content)
    print(f"Download: {name}")
    

def main(args):
    download_yolov5_weight(args.dst_folder, "yolov5m.pt")
    download_yolov5_weight(args.dst_folder, "yolov5l.pt")
    download_yolov5_weight(args.dst_folder, "yolov5s6.pt")
    download_yolov5_weight(args.dst_folder, "yolov5m6.pt")

#     download_yolov7_weight(args.dst_folder, "yolov7.pt")
#     download_yolov7_weight(args.dst_folder, "yolov7x.pt")
#     download_yolov7_weight(args.dst_folder, "yolov7-w6.pt")


if __name__ == "__main__":
    args = get_args()
    
    # make dir
    os.makedirs(args.dst_folder, exist_ok=True)

    # main
    main(args)