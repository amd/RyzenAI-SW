"""
This module prepares the COCO dataset by downloading and converting annotations.
"""

import os
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
import wget
import numpy as np
from tqdm import tqdm


def download_and_extract(url, destination):
    print(f"Downloading from {url}...")
    filename = wget.download(url, out=destination)
    print(f"\nExtracting {filename}...")
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(destination)
    os.remove(filename)
    print(f"Extraction complete: {destination}")


def make_dirs(directory="./datasets/coco"):
    dir_path = Path(directory)
    (dir_path / "labels").mkdir(parents=True, exist_ok=True)
    return dir_path


def coco91_to_coco80_class():
    """
    Maps COCO's 91-class IDs to 80-class IDs.

    Returns:
        list: A mapping list from 91-class to 80-class.
    """
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]


def convert_coco_json(
    json_dir="./datasets/coco/annotations/", use_segments=False, cls91to80=False
):
    """
    Converts COCO JSON annotations to YOLO format.

    Args:
        json_dir (str): Directory containing COCO JSON files.
        use_segments (bool): Whether to use segmentation data.
        cls91to80 (bool): Whether to convert 91-class to 80-class.
    """
    save_dir = make_dirs()
    coco80 = coco91_to_coco80_class()

    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        if not str(json_file).endswith("instances_val2017.json"):
            continue

        fn = Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")
        fn.mkdir(parents=True, exist_ok=True)
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        images = {"%g" % x["id"]: x for x in data["images"]}
        img_to_anns = defaultdict(list)
        for ann in data["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        with open(
            Path(save_dir / "val2017").with_suffix(".txt"), "a", encoding="utf-8"
        ) as txt_file:
            for img_id, anns in tqdm(
                img_to_anns.items(), desc=f"Annotations {json_file}"
            ):
                img = images["%g" % img_id]
                h, w, f = img["height"], img["width"], img["file_name"]
                bboxes = []
                segments = []

                txt_file.write(
                    f"./images/{'/'.join(img['coco_url'].split('/')[-2:])}\n"
                )
                for ann in anns:
                    if ann["iscrowd"]:
                        continue
                    box = np.array(ann["bbox"], dtype=np.float64)
                    box[:2] += box[2:] / 2  # xy top-left corner to center
                    box[[0, 2]] /= w  # normalize x
                    box[[1, 3]] /= h  # normalize y
                    if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                        continue

                    cls = (
                        coco80[ann["category_id"] - 1]
                        if cls91to80
                        else ann["category_id"] - 1
                    )
                    box = [cls] + box.tolist()
                    if box not in bboxes:
                        bboxes.append(box)

                    if use_segments and ann.get("segmentation"):
                        seg = ann["segmentation"]
                        if isinstance(seg, list) and len(seg) > 0:
                            s = np.array(seg[0]).reshape(-1, 2)
                            s[:, 0] /= w  # normalize x
                            s[:, 1] /= h  # normalize y
                            s = [cls] + s.reshape(-1).tolist()
                            if s not in segments:
                                segments.append(s)

                with open((fn / f).with_suffix(".txt"), "a", encoding="utf-8") as file:
                    for i, bbox in enumerate(bboxes):
                        line = (
                            segments[i] if use_segments and i < len(segments) else bbox
                        )
                        file.write(("%g " * len(line)).rstrip() % tuple(line) + "\n")


def main():
    """
    Main function to download COCO dataset and convert annotations.
    """
    base_dir = "./datasets/coco"
    os.makedirs(base_dir, exist_ok=True)

    # Download and extract images
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    download_and_extract("http://images.cocodataset.org/zips/val2017.zip", images_dir)

    # Download and extract annotations
    annotations_dir = os.path.join(base_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    download_and_extract(
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        annotations_dir,
    )

    # Copy instances_val2017.json to parent folder
    shutil.copy(
        os.path.join(annotations_dir, "annotations", "instances_val2017.json"),
        os.path.join(annotations_dir, "instances_val2017.json"),
    )

    # Convert annotations
    print("Converting COCO JSON annotations to YOLO format...")
    convert_coco_json(
        json_dir=os.path.join(annotations_dir, "annotations"),
        use_segments=True,
        cls91to80=True,
    )

    print("COCO dataset preparation completed.")


if __name__ == "__main__":
    main()
