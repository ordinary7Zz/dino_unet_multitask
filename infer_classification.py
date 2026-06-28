#!/usr/bin/env python3
"""
推理脚本：使用 DINOv3_S_UNet_MULTITASK 的分类头对图像目录进行批量推理。
输出 CSV 文件，包含文件名、预测类别和各类别概率/置信度分数。
"""

import os
import argparse
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm

from dino_unet_multitask import DINOv3_S_UNet_MULTITASK


# ---------------------------------------------------------------------------
# 自定义推理 Dataset（不需要 mask，直接使用分类预处理）
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """纯分类推理用 Dataset：扫描目录中所有图像文件。"""

    VALID_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    def __init__(self, image_dir: str, img_size: int = 224):
        self.image_dir = image_dir
        self.img_size = img_size
        self.image_paths = sorted(
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(self.VALID_SUFFIXES)
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No image files found in {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        filename = os.path.basename(path)
        with Image.open(path) as img:
            img = img.convert('RGB')
        tensor = self.transform(img)
        return tensor, filename


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def str2bool(value: str) -> bool:
    return str(value).lower() in ('true', '1', 'yes', 'y')


def load_model(checkpoint_path: str, pretrained: bool, use_dilation: bool,
               device: torch.device) -> DINOv3_S_UNet_MULTITASK:
    """加载模型并恢复权重。"""
    model = DINOv3_S_UNet_MULTITASK(pretrained=pretrained, use_dilation=use_dilation)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 推理主流程
# ---------------------------------------------------------------------------

def main(args):
    # ---------- 检查参数 ----------
    if args.num_classes not in (2, 5):
        raise ValueError(f"num_classes must be 2 or 5, got {args.num_classes}")

    # ---------- 设备 ----------
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- 加载模型 ----------
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.dino_pretrained, args.use_dilation, device)

    # ---------- 构建数据加载器 ----------
    dataset = InferenceDataset(args.image_dir, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    # ---------- 推理 ----------
    csv_rows = []

    with torch.no_grad():
        for images, filenames in tqdm(loader, desc="Inference"):
            images = images.to(device)

            # 模型前向（只关注分类输出）
            _, benign_malignant, tirads = model(images)

            if args.num_classes == 2:
                # 二分类：sigmoid → 概率；threshold 0.5 → 类别
                logits = benign_malignant.squeeze(1)          # (B,)
                probs_1 = torch.sigmoid(logits)               # P(class=1)
                probs_0 = 1.0 - probs_1
                preds = (probs_1 > 0.5).long()

                for i, fname in enumerate(filenames):
                    csv_rows.append({
                        'filename': fname,
                        'predicted_class': int(preds[i].item()),
                        'prob_0': float(probs_0[i].item()),
                        'prob_1': float(probs_1[i].item()),
                    })
            else:
                # 五分类：softmax → 概率分布；argmax → 类别
                probs = F.softmax(tirads, dim=1)               # (B, 5)
                preds = probs.argmax(dim=1)

                for i, fname in enumerate(filenames):
                    row = {
                        'filename': fname,
                        'predicted_class': int(preds[i].item()),
                    }
                    for c in range(args.num_classes):
                        row[f'prob_{c}'] = float(probs[i, c].item())
                    csv_rows.append(row)

    # ---------- 写出 CSV ----------
    print(f"Saving results to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    fieldnames = ['filename', 'predicted_class'] + [f'prob_{c}' for c in range(args.num_classes)]
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Done. {len(csv_rows)} images processed.")
    # 打印类别分布统计
    from collections import Counter
    counter = Counter(row['predicted_class'] for row in csv_rows)
    print(f"Predicted class distribution: {dict(sorted(counter.items()))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DINOv3-UNet Multitask Classification Inference"
    )

    # 必填
    parser.add_argument("--image_dir", type=str, required=True,
                        help="待推理图像所在目录")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重文件路径 (.pth)")
    parser.add_argument("--num_classes", type=int, required=True,
                        choices=[2, 5],
                        help="分类类别数 (2: 良恶性二分类, 5: TIRADS五分类)")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 CSV 文件路径")

    # 图像与模型配置（需与训练时一致）
    parser.add_argument("--img_size", type=int, default=224,
                        help="输入图像尺寸 (默认: 224)")
    parser.add_argument("--dino_pretrained", type=str, default='True',
                        help="DINO backbone 是否使用预训练权重 (True/False)")
    parser.add_argument("--use_dilation", type=str, default='False',
                        help="模型是否使用 dilation 层 (True/False)")

    # 硬件
    parser.add_argument("--cuda_device", type=int, default=0,
                        help="CUDA 设备索引 (默认: 0)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="推理批大小 (默认: 16)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader 子进程数 (默认: 4)")

    args = parser.parse_args()

    # 布尔类型转换
    args.dino_pretrained = str2bool(args.dino_pretrained)
    args.use_dilation = str2bool(args.use_dilation)

    main(args)
