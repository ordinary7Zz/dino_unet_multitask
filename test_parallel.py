import argparse
import json
import os
import sys
import torch
import numpy as np
import time
import logging
import imageio
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import evaluate_model, compute_youden_threshold
from utils.utils import gla_params, log_print
from dataset import MultiTaskDataset
from dino_unet_multitask import DINOv3_S_UNet_MULTITASK

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def clean_path(path):
    """Clean path by removing extra quotes and whitespace."""
    if isinstance(path, str):
        # Remove quotes if present
        if (path.startswith('"') and path.endswith('"')) or \
           (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        # Strip whitespace
        path = path.strip()
    return path


def process_dataset(
    model,
    image_size,
    threshold_info,
    image_path,
    gt_path,
    label_path,
    save_base_path,
    dataset_name,
    device,
    save_results,
    malignancy_threshold: float,
    use_fixed_threshold: bool = False,
):
    # Create save directory for this dataset
    save_path = os.path.join(save_base_path, dataset_name)
    if save_results.lower() == "true":
        os.makedirs(save_path, exist_ok=True)

    # Additional path cleaning
    image_path = clean_path(image_path)
    gt_path = clean_path(gt_path)
    if label_path:
        label_path = clean_path(label_path)

    log_print(f"Processing dataset: {dataset_name}")
    log_print(f"Image path: {image_path}")
    log_print(f"GT path: {gt_path}")
    log_print(f"Label path: {label_path if label_path else 'None'}")
    log_print(f"Save path: {save_path}")

    # Check path existence before loading dataset
    _empty_result = lambda name: {
        "dataset": name,
        "Dice": {"mean": 0.0, "CI95": (0.0, 0.0)},
        "HD95": {"mean": 0.0, "CI95": (0.0, 0.0)},
        "Malignancy": {},
        "TIRADS": {},
    }
    if not os.path.exists(image_path):
        print(f"Error: Image directory does not exist: {image_path}")
        return _empty_result(dataset_name)

    if not os.path.exists(gt_path):
        print(f"Error: Mask directory does not exist: {gt_path}")
        return _empty_result(dataset_name)

    # List directory contents for debugging
    try:
        image_files = os.listdir(image_path)
        mask_files = os.listdir(gt_path)
        print(f"Found {len(image_files)} files in image directory")
        print(f"Found {len(mask_files)} files in mask directory")
    except Exception as e:
        print(f"Error listing directory contents: {e}")

    # 加载测试数据集，使用与训练时相同的尺寸
    target_size = image_size
    print(f"Loading dataset with MultiTaskDataset using size: {target_size}x{target_size}")
    test_dataset = MultiTaskDataset(image_path, gt_path, label_path, target_size, mode='test')

    # 检查数据集是否为空
    if len(test_dataset) == 0:
        print(f"Error: Test dataset {dataset_name} is empty!")
        return _empty_result(dataset_name)

    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=12, num_workers=8)
    print(f"Dataset loaded: {len(test_dataset)} images found")

    # 计算评估指标 (使用 utils.new_metrics.evaluate_model 的返回结构)
    print(f"Calculating evaluation metrics for dataset: {dataset_name}")
    print(f"Using malignancy threshold (from val): {malignancy_threshold}")

    results = evaluate_model(
        model,
        test_loader,
        device,
        threshold=malignancy_threshold,
    )

    # Segmentation metrics
    dice_info = results.get('Dice', {})
    hd95_info = results.get('HD95', {})

    dice_mean = dice_info.get('mean', float('nan'))
    dice_ci = dice_info.get('CI95', (float('nan'), float('nan')))
    hd95_mean = hd95_info.get('mean', float('nan'))
    hd95_ci = hd95_info.get('CI95', (float('nan'), float('nan')))

    print(
        f"Dice: mean={dice_mean:.4f}, CI95=({dice_ci[0]:.4f}, {dice_ci[1]:.4f})"
    )
    print(
        f"HD95: mean={hd95_mean:.4f}, CI95=({hd95_ci[0]:.4f}, {hd95_ci[1]:.4f})"
    )
    if use_fixed_threshold:
        print(f"Malignancy Threshold (fixed): {threshold_info.get('best_threshold', float('nan')):.4f}")
    else:
        print(
            f"Youden Best Threshold (from val): {threshold_info.get('best_threshold', float('nan'))}, "
            f"Youden (val): {threshold_info.get('youden', float('nan'))}, "
            f"Sensitivity (val): {threshold_info.get('sensitivity', float('nan'))}, "
            f"Specificity (val): {threshold_info.get('specificity', float('nan'))}"
        )

    # Malignancy / TIRADS 分类结果按照 utils.new_metrics 返回结构组织
    malignancy_metrics = results.get('Malignancy', {})
    tirads_metrics = results.get('TIRADS', {})

    # 阈值信息：使用传入的 malignancy_threshold
    print(f"Malignancy Used Threshold (on test): {float(malignancy_threshold):.4f}")

    # 输出 malignancy 指标的 mean 和 CI95
    if isinstance(malignancy_metrics, dict) and len(malignancy_metrics) > 0:
        print("Malignancy Metrics:")
        for k, v in sorted(malignancy_metrics.items()):
            if isinstance(v, dict):
                mean_v = v.get('mean', float('nan'))
                ci_v = v.get('CI95', (float('nan'), float('nan')))
                print(f" - {k}: mean={mean_v:.4f}, CI95=({ci_v[0]:.4f}, {ci_v[1]:.4f})")
            else:
                # fallback if structure unexpected
                try:
                    val = float(v)
                    print(f" - {k}: {val:.4f}")
                except Exception:
                    print(f" - {k}: {v}")

    # 输出 TIRADS 指标的 mean 和 CI95
    if isinstance(tirads_metrics, dict) and len(tirads_metrics) > 0:
        print("TIRADS Metrics:")
        for k, v in sorted(tirads_metrics.items()):
            if isinstance(v, dict):
                mean_v = v.get('mean', float('nan'))
                ci_v = v.get('CI95', (float('nan'), float('nan')))
                print(f" - {k}: mean={mean_v:.4f}, CI95=({ci_v[0]:.4f}, {ci_v[1]:.4f})")
            else:
                try:
                    val = float(v)
                    print(f" - {k}: {val:.4f}")
                except Exception:
                    print(f" - {k}: {v}")

    # 保存预测结果（如果需要）
    if save_results.lower() == "true":
        model.eval()
        for i, batch in enumerate(tqdm(test_loader, desc='Saving predictions', unit='image')):
            with torch.no_grad():
                image = batch['image'].to(device=device)
                name = batch.get('filename', [f'image_{i}'])[0]

                # Forward pass
                outputs = model(image)
                # 获取分割输出
                if isinstance(outputs, (list, tuple)):
                    res = outputs[0]  # 分割结果
                else:
                    res = outputs

                # 后处理和保存
                res_sigmoid = res.sigmoid().data.cpu()
                res_np = res_sigmoid.numpy().squeeze()
                res_normalized = (res_np - res_np.min()) / (res_np.max() - res_np.min() + 1e-8)
                res_uint8 = (res_normalized * 255).astype(np.uint8)

                # 使用从数据集中获取的原始文件名（不包含扩展名）
                output_filename = f"{name}.png"
                try:
                    imageio.imsave(os.path.join(save_path, output_filename), res_uint8)
                except Exception as e:
                    print(f"Error saving prediction for {name}: {e}")

    print(f"Dataset {dataset_name} processing completed.")

    # 返回完整结果，包含分割与分类指标的 mean 和 CI95
    return {
        "dataset": dataset_name,
        "Dice": {"mean": dice_mean, "CI95": dice_ci},
        "HD95": {"mean": hd95_mean, "CI95": hd95_ci},
        "Malignancy": malignancy_metrics,
        "TIRADS": tirads_metrics,
    }


def _print_summary(all_results, log_file):
    """打印汇总表并保存 JSON，每个分类指标均含 mean 和 CI95。"""
    if not all_results:
        return

    print("\n" + "=" * 80)
    print("SUMMARY: All Datasets (with CI95 for each metric)")
    print("=" * 80)

    for res in all_results:
        ds = res.get("dataset", "?")
        print(f"\n--- {ds} ---")
        for key in ["Dice", "HD95"]:
            v = res.get(key, {})
            mean_v = v.get("mean", float("nan"))
            ci = v.get("CI95", (float("nan"), float("nan")))
            print(f"  {key}: mean={mean_v:.4f}, CI95=({ci[0]:.4f}, {ci[1]:.4f})")

        mal = res.get("Malignancy", {})
        if mal:
            print("  Malignancy:")
            for k, v in sorted(mal.items()):
                if isinstance(v, dict):
                    mean_v = v.get("mean", float("nan"))
                    ci = v.get("CI95", (float("nan"), float("nan")))
                    print(f"    {k}: mean={mean_v:.4f}, CI95=({ci[0]:.4f}, {ci[1]:.4f})")

        tir = res.get("TIRADS", {})
        if tir:
            print("  TIRADS:")
            for k, v in sorted(tir.items()):
                if isinstance(v, dict):
                    mean_v = v.get("mean", float("nan"))
                    ci = v.get("CI95", (float("nan"), float("nan")))
                    print(f"    {k}: mean={mean_v:.4f}, CI95=({ci[0]:.4f}, {ci[1]:.4f})")

    # 保存 JSON（便于程序化读取）
    json_path = log_file.replace(".log", "_results.json")
    try:
        def _to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {kk: _to_serializable(vv) for kk, vv in obj.items()}
            return obj

        json_results = [_to_serializable(res) for res in all_results]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {json_path}")
    except Exception as e:
        print(f"Warning: Could not save JSON results: {e}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser("DINOV3-UNet Test (Youden Threshold from Val)")
    # 仅保留一套主参数名，去掉多余别名与 help
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_image_paths", type=str, action="append", default=[])
    parser.add_argument("--test_gt_paths", type=str, action="append", default=[])
    parser.add_argument("--test_label_paths", type=str, action="append", default=[])
    parser.add_argument("--test_dataset_names", type=str, action="append", default=[])
    parser.add_argument("--threshold_malignancy", type=float, default=None,
                        help="若设置则直接使用该阈值，不再用 Youden 计算；未设置时需提供 val_* 路径")
    parser.add_argument("--val_image_path", type=str, default=None)
    parser.add_argument("--val_gt_path", type=str, default=None)
    parser.add_argument("--val_label_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./predictions")
    parser.add_argument("--save_results", type=str, default="true")
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='CUDA device index to use (default: 0)')
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--img_size", type=int, default=224,
                        help="input image size for dataset (default: 224)")
    parser.add_argument('--dino_pretrained', type=str, default='True',
                        help='whether to load pretrained weights for the DINO backbone (True/False)')
    parser.add_argument('--use_dilation', type=str, default='False',
                        help='whether to use dilation layers in the model (True/False, default: False)')
    args = parser.parse_args()
    args.dino_pretrained = str(args.dino_pretrained).lower() in ('true', '1', 'yes', 'y')
    args.use_dilation = str(args.use_dilation).lower() in ('true', '1', 'yes', 'y')

    # 未设置 threshold_malignancy 时，必须提供 val 路径
    if args.threshold_malignancy is None:
        if not args.val_image_path or not args.val_gt_path or not args.val_label_path:
            parser.error("--val_image_path, --val_gt_path, --val_label_path are required when --threshold_malignancy is not set")

    # Configure logging system
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"test_youden_threshold_{timestamp}.log")

    # Set up logger
    sys.stdout = Logger(log_file)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"CUDA visible devices: {args.cuda_device}")
    print(f"Image size: {args.img_size} x {args.img_size}")
    print(f"DINO pretrained: {args.dino_pretrained}")
    print(f"DINO Dilation: {args.use_dilation}")
    
    # Log configuration
    print(f"Checkpoint path: {args.checkpoint}")
    print(f"Save path: {args.save_path}")
    print(f"Save results: {args.save_results}")
    print(f"Test dataset names: {args.test_dataset_names}")
    if args.threshold_malignancy is not None:
        print(f"Threshold malignancy (fixed): {args.threshold_malignancy}")
    else:
        print(f"Val image path: {args.val_image_path}")
        print(f"Val gt path: {args.val_gt_path}")
        print(f"Val label path: {args.val_label_path}")

    # Load model once
    print("Loading model...")
    model = DINOv3_S_UNet_MULTITASK(pretrained=args.dino_pretrained, use_dilation=args.use_dilation).to(device)
    checkpoint_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint_dict, strict=False)
    print(f"Successfully loaded checkpoint from {args.checkpoint}\n")

    # 确定 malignancy 阈值：优先使用 --threshold_malignancy，否则用 Youden 计算
    if args.threshold_malignancy is not None:
        best_threshold = float(args.threshold_malignancy)
        threshold_info = {
            'best_threshold': best_threshold,
            'youden': float('nan'),
            'sensitivity': float('nan'),
            'specificity': float('nan'),
        }
        print(f"Using fixed malignancy threshold: {best_threshold:.4f} (--threshold_malignancy)\n")
    else:
        args.val_image_path = clean_path(args.val_image_path)
        args.val_gt_path = clean_path(args.val_gt_path)
        args.val_label_path = clean_path(args.val_label_path)
        print(f"Loading validation dataset with MultiTaskDataset using size: {args.img_size}x{args.img_size}")
        val_dataset = MultiTaskDataset(args.val_image_path, args.val_gt_path, args.val_label_path, args.img_size, mode='test')
        if len(val_dataset) == 0:
            print("Error: Validation dataset is empty!")
            return
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=12, num_workers=8)
        print(f"Validation dataset loaded: {len(val_dataset)} images found")
        print("Computing malignancy threshold on validation set (Youden)...")
        threshold_info = compute_youden_threshold(model, val_loader, device)
        best_threshold = float(threshold_info.get('best_threshold', 0.5))
        print(
            f"Youden Best Threshold (val): {threshold_info.get('best_threshold', float('nan'))}, "
            f"Youden: {threshold_info.get('youden', float('nan'))}, "
            f"Sensitivity: {threshold_info.get('sensitivity', float('nan'))}, "
            f"Specificity: {threshold_info.get('specificity', float('nan'))}\n"
        )

    # Clean test paths by removing any extra quotes
    args.test_image_paths = [clean_path(path) for path in args.test_image_paths]
    args.test_gt_paths = [clean_path(path) for path in args.test_gt_paths]
    if args.test_label_paths:
        args.test_label_paths = [clean_path(path) if path else None for path in args.test_label_paths]
    else:
        # 如果未提供标签路径,使用None填充
        args.test_label_paths = [None] * len(args.test_image_paths)

    # Ensure test image paths and mask paths数量匹配
    if len(args.test_image_paths) != len(args.test_gt_paths):
        print(
            f"Warning: Number of test image paths ({len(args.test_image_paths)}) and mask paths ({len(args.test_gt_paths)}) do not match."
        )
        print("Using the minimum number of pairs.")
        min_len = min(len(args.test_image_paths), len(args.test_gt_paths))
        args.test_image_paths = args.test_image_paths[:min_len]
        args.test_gt_paths = args.test_gt_paths[:min_len]

    # Create base save directory
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Created base save directory: {args.save_path}")

    # Process each test dataset and collect results (including CI95 for each metric)
    start_time = time.time()
    all_results = []
    for i, (img_path, gt_path) in enumerate(zip(args.test_image_paths, args.test_gt_paths)):
        # Use provided dataset name if available, otherwise use default naming
        if i < len(args.test_dataset_names) and args.test_dataset_names[i]:
            dataset_name = args.test_dataset_names[i]
        else:
            dataset_name = f"Test_Set_{i+1}"

        # Get label path if available
        label_path = args.test_label_paths[i] if i < len(args.test_label_paths) else None

        print(f"\nProcessing dataset {i+1}/{len(args.test_image_paths)}")
        dataset_result = process_dataset(
            model,
            args.img_size,
            threshold_info,
            img_path,
            gt_path,
            label_path,
            args.save_path,
            dataset_name,
            device,
            args.save_results,
            malignancy_threshold=best_threshold,
            use_fixed_threshold=(args.threshold_malignancy is not None),
        )
        all_results.append(dataset_result)

    total_time = time.time() - start_time
    print(f"All datasets processed in {total_time:.2f} seconds")

    # 打印汇总表，每个分类指标均含 mean 和 CI95
    _print_summary(all_results, log_file)

    # Close logger properly
    try:
        if hasattr(sys.stdout, 'close') and hasattr(sys.stdout, '_closed') and not sys.stdout._closed:
            sys.stdout.close()
    except Exception:
        # Just handle the exception to avoid script failure
        pass


if __name__ == "__main__":
    main()
