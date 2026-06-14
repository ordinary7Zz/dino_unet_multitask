# 本脚本用于根据分类标签文件，将图像按指定标签值复制到对应的 0/1 目录下，并重命名后保存

import argparse
import json
import shutil
from pathlib import Path


def load_records(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "filename" in data:
        return [data]
    if isinstance(data, dict):
        for key in ("data", "items", "records", "annotations"):
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("无法识别的 JSON 格式，请确认是样本列表或单条样本格式。")


def copy_images_by_label(json_file, label_name, image_root, save_root):
    json_file = Path(json_file)
    image_root = Path(image_root)
    save_root = Path(save_root)

    records = load_records(json_file)

    for idx, item in enumerate(records):
        if "filename" not in item:
            print(f"[跳过] 第 {idx} 条没有 filename")
            continue

        if label_name not in item:
            raise KeyError(f"第 {idx} 条记录中不存在标签: {label_name}")

        label_value = int(item[label_name])
        if label_value not in (0, 1):
            print(f"[跳过] 第 {idx} 条标签值不是 0/1: {label_value}")
            continue

        rel_path = Path(item["filename"])
        src_path = image_root / rel_path

        if not src_path.exists():
            print(f"[缺失] 源文件不存在: {src_path}")
            continue

        new_name = "_".join(rel_path.parts)
        dst_path = save_root / str(label_value) / new_name
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_path, dst_path)
        print(f"[复制] {src_path} -> {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", required=True, help="分类标签 json 文件路径")
    parser.add_argument("--label_name", required=True, help="要筛选的标签名，如 LNM_CN01")
    parser.add_argument("--image_root", required=True, help="图像根目录")
    parser.add_argument("--save_root", required=True, help="保存目录")

    args = parser.parse_args()

    copy_images_by_label(
        json_file=args.json_file,
        label_name=args.label_name,
        image_root=args.image_root,
        save_root=args.save_root,
    )
