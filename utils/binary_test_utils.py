import json
import os

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MultiTaskDataset
from utils.metrics import evaluate_model_binary_target
from utils.utils import log_print


BINARY_BATCH_SIZE = 12
BINARY_NUM_WORKERS = 8


def clean_path(path):
    if isinstance(path, str):
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        path = path.strip()
    return path


def build_empty_binary_result(dataset_name, target_key):
    return {
        'dataset': dataset_name,
        'Dice': {'mean': 0.0, 'CI95': (0.0, 0.0)},
        'HD95': {'mean': 0.0, 'CI95': (0.0, 0.0)},
        target_key: {},
    }


def process_binary_dataset(
    model,
    image_size,
    threshold_info,
    image_path,
    gt_path,
    label_path,
    save_base_path,
    dataset_name,
    target_key,
    device,
    save_results,
    threshold,
    use_fixed_threshold=False,
):
    save_path = os.path.join(save_base_path, dataset_name)
    if save_results.lower() == 'true':
        os.makedirs(save_path, exist_ok=True)

    image_path = clean_path(image_path)
    gt_path = clean_path(gt_path)
    label_path = clean_path(label_path) if label_path else None

    log_print(f"Processing dataset: {dataset_name}\n")
    log_print(f"Image path: {image_path}\n")
    log_print(f"GT path: {gt_path}\n")
    log_print(f"Label path: {label_path if label_path else 'None'}\n")
    log_print(f"Save path: {save_path}\n")

    if not os.path.exists(image_path):
        print(f"Error: Image directory does not exist: {image_path}")
        return build_empty_binary_result(dataset_name, target_key)

    if not os.path.exists(gt_path):
        print(f"Error: Mask directory does not exist: {gt_path}")
        return build_empty_binary_result(dataset_name, target_key)

    try:
        image_files = os.listdir(image_path)
        mask_files = os.listdir(gt_path)
        print(f"Found {len(image_files)} files in image directory")
        print(f"Found {len(mask_files)} files in mask directory")
    except Exception as e:
        print(f"Error listing directory contents: {e}")

    print(f"Loading dataset with MultiTaskDataset using size: {image_size}x{image_size}")
    test_dataset = MultiTaskDataset(
        image_path,
        gt_path,
        label_path,
        image_size,
        mode='test',
        target_key=target_key,
    )

    if len(test_dataset) == 0:
        print(f"Error: Test dataset {dataset_name} is empty!")
        return build_empty_binary_result(dataset_name, target_key)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=BINARY_BATCH_SIZE,
        num_workers=BINARY_NUM_WORKERS,
    )
    print(f"Dataset loaded: {len(test_dataset)} images found")

    print(f"Calculating evaluation metrics for dataset: {dataset_name}")
    print(f"Using {target_key} threshold: {threshold}")

    results = evaluate_model_binary_target(
        model,
        test_loader,
        device,
        threshold=threshold,
        target_field='target',
        target_name=target_key,
    )

    dice_info = results.get('Dice', {})
    hd95_info = results.get('HD95', {})
    cls_metrics = results.get(target_key, {})

    dice_mean = dice_info.get('mean', float('nan'))
    dice_ci = dice_info.get('CI95', (float('nan'), float('nan')))
    hd95_mean = hd95_info.get('mean', float('nan'))
    hd95_ci = hd95_info.get('CI95', (float('nan'), float('nan')))

    print(f"Dice: mean={dice_mean:.4f}, CI95=({dice_ci[0]:.4f}, {dice_ci[1]:.4f})")
    print(f"HD95: mean={hd95_mean:.4f}, CI95=({hd95_ci[0]:.4f}, {hd95_ci[1]:.4f})")

    if use_fixed_threshold:
        print(f"{target_key} Threshold (fixed): {threshold_info.get('best_threshold', float('nan')):.4f}")
    else:
        print(
            f"Youden Best Threshold (from val): {threshold_info.get('best_threshold', float('nan'))}, "
            f"Youden (val): {threshold_info.get('youden', float('nan'))}, "
            f"Sensitivity (val): {threshold_info.get('sensitivity', float('nan'))}, "
            f"Specificity (val): {threshold_info.get('specificity', float('nan'))}"
        )

    print(f"{target_key} Used Threshold (on test): {float(threshold):.4f}")

    if isinstance(cls_metrics, dict) and cls_metrics:
        print(f"{target_key} Metrics:")
        for k, v in sorted(cls_metrics.items()):
            if isinstance(v, dict):
                mean_v = v.get('mean', float('nan'))
                ci_v = v.get('CI95', (float('nan'), float('nan')))
                print(f" - {k}: mean={mean_v:.4f}, CI95=({ci_v[0]:.4f}, {ci_v[1]:.4f})")
            else:
                try:
                    print(f" - {k}: {float(v):.4f}")
                except Exception:
                    print(f" - {k}: {v}")

    if save_results.lower() == 'true':
        model.eval()
        for i, batch in enumerate(tqdm(test_loader, desc='Saving predictions', unit='image')):
            with torch.no_grad():
                image = batch['image'].to(device=device)
                name = batch.get('filename', [f'image_{i}'])[0]
                outputs = model(image)
                res = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                res_sigmoid = res.sigmoid().data.cpu()
                res_np = res_sigmoid.numpy().squeeze()
                res_normalized = (res_np - res_np.min()) / (res_np.max() - res_np.min() + 1e-8)
                res_uint8 = (res_normalized * 255).astype(np.uint8)
                output_filename = f"{name}.png"
                try:
                    imageio.imsave(os.path.join(save_path, output_filename), res_uint8)
                except Exception as e:
                    print(f"Error saving prediction for {name}: {e}")

    print(f"Dataset {dataset_name} processing completed.")

    return {
        'dataset': dataset_name,
        'Dice': {'mean': dice_mean, 'CI95': dice_ci},
        'HD95': {'mean': hd95_mean, 'CI95': hd95_ci},
        target_key: cls_metrics,
    }


def print_binary_summary(all_results, log_file, target_key):
    if not all_results:
        return

    print("\n" + "=" * 80)
    print("SUMMARY: All Datasets (with CI95 for each metric)")
    print("=" * 80)

    for res in all_results:
        ds = res.get('dataset', '?')
        print(f"\n--- {ds} ---")
        for key in ['Dice', 'HD95']:
            v = res.get(key, {})
            mean_v = v.get('mean', float('nan'))
            ci = v.get('CI95', (float('nan'), float('nan')))
            print(f"  {key}: mean={mean_v:.4f}, CI95=({ci[0]:.4f}, {ci[1]:.4f})")

        cls_metrics = res.get(target_key, {})
        if cls_metrics:
            print(f"  {target_key}:")
            for k, v in sorted(cls_metrics.items()):
                if isinstance(v, dict):
                    mean_v = v.get('mean', float('nan'))
                    ci = v.get('CI95', (float('nan'), float('nan')))
                    print(f"    {k}: mean={mean_v:.4f}, CI95=({ci[0]:.4f}, {ci[1]:.4f})")

    json_path = log_file.replace('.log', '_results.json')
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
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {json_path}")
    except Exception as e:
        print(f"Warning: Could not save JSON results: {e}")
