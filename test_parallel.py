import argparse
import json
import os
import sys
import time
from datetime import datetime

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import evaluate_model, compute_youden_threshold
from dataset import MultiTaskDataset
from dino_unet_multitask import DINOv3_S_UNet_MULTITASK


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def clean_path(path):
    if isinstance(path, str):
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
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
    save_path = os.path.join(save_base_path, dataset_name)
    if save_results.lower() == 'true':
        os.makedirs(save_path, exist_ok=True)

    image_path = clean_path(image_path)
    gt_path = clean_path(gt_path)
    label_path = clean_path(label_path) if label_path else None

    print(f'Processing dataset: {dataset_name}')
    print(f'Image path: {image_path}')
    print(f'GT path: {gt_path}')
    print(f'Label path: {label_path if label_path else "None"}')
    print(f'Save path: {save_path}')

    def _empty_result(name):
        return {
            'dataset': name,
            'Dice': {'mean': 0.0, 'CI95': (0.0, 0.0)},
            'HD95': {'mean': 0.0, 'CI95': (0.0, 0.0)},
            'Malignancy': {},
            'TIRADS': {},
        }

    if not os.path.exists(image_path):
        print(f'Error: Image directory does not exist: {image_path}')
        return _empty_result(dataset_name)

    if not os.path.exists(gt_path):
        print(f'Error: Mask directory does not exist: {gt_path}')
        return _empty_result(dataset_name)

    try:
        image_files = os.listdir(image_path)
        mask_files = os.listdir(gt_path)
        print(f'Found {len(image_files)} files in image directory')
        print(f'Found {len(mask_files)} files in mask directory')
    except Exception as e:
        print(f'Error listing directory contents: {e}')

    print(f'Loading dataset with MultiTaskDataset using size: {image_size}x{image_size}')
    test_dataset = MultiTaskDataset(image_path, gt_path, label_path, image_size, mode='test')
    if len(test_dataset) == 0:
        print(f'Error: Test dataset {dataset_name} is empty!')
        return _empty_result(dataset_name)

    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=12, num_workers=8)
    print(f'Dataset loaded: {len(test_dataset)} images found')
    print(f'Calculating evaluation metrics for dataset: {dataset_name}')
    print(f'Using Malignancy threshold: {malignancy_threshold}')

    results = evaluate_model(model, test_loader, device, threshold=malignancy_threshold)

    dice_info = results.get('Dice', {})
    hd95_info = results.get('HD95', {})
    malignancy_metrics = results.get('Malignancy', {})
    tirads_metrics = results.get('TIRADS', {})

    dice_mean = dice_info.get('mean', float('nan'))
    dice_ci = dice_info.get('CI95', (float('nan'), float('nan')))
    hd95_mean = hd95_info.get('mean', float('nan'))
    hd95_ci = hd95_info.get('CI95', (float('nan'), float('nan')))

    print(f'Dice: mean={dice_mean:.4f}, CI95=({dice_ci[0]:.4f}, {dice_ci[1]:.4f})')
    print(f'HD95: mean={hd95_mean:.4f}, CI95=({hd95_ci[0]:.4f}, {hd95_ci[1]:.4f})')

    if use_fixed_threshold:
        print(f"Malignancy Threshold (fixed): {threshold_info.get('best_threshold', float('nan')):.4f}")
    else:
        print(
            f"Youden Best Threshold (from val): {threshold_info.get('best_threshold', float('nan'))}, "
            f"Youden (val): {threshold_info.get('youden', float('nan'))}, "
            f"Sensitivity (val): {threshold_info.get('sensitivity', float('nan'))}, "
            f"Specificity (val): {threshold_info.get('specificity', float('nan'))}"
        )

    print(f'Malignancy Used Threshold (on test): {float(malignancy_threshold):.4f}')

    if isinstance(malignancy_metrics, dict) and malignancy_metrics:
        print('Malignancy Metrics:')
        for k, v in sorted(malignancy_metrics.items()):
            if isinstance(v, dict):
                mean_v = v.get('mean', float('nan'))
                ci_v = v.get('CI95', (float('nan'), float('nan')))
                print(f' - {k}: mean={mean_v:.4f}, CI95=({ci_v[0]:.4f}, {ci_v[1]:.4f})')

    if isinstance(tirads_metrics, dict) and tirads_metrics:
        print('TIRADS Metrics:')
        for k, v in sorted(tirads_metrics.items()):
            if isinstance(v, dict):
                mean_v = v.get('mean', float('nan'))
                ci_v = v.get('CI95', (float('nan'), float('nan')))
                print(f' - {k}: mean={mean_v:.4f}, CI95=({ci_v[0]:.4f}, {ci_v[1]:.4f})')

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
                output_filename = f'{name}.png'
                try:
                    imageio.imsave(os.path.join(save_path, output_filename), res_uint8)
                except Exception as e:
                    print(f'Error saving prediction for {name}: {e}')

    print(f'Dataset {dataset_name} processing completed.')

    return {
        'dataset': dataset_name,
        'Dice': {'mean': dice_mean, 'CI95': dice_ci},
        'HD95': {'mean': hd95_mean, 'CI95': hd95_ci},
        'Malignancy': malignancy_metrics,
        'TIRADS': tirads_metrics,
    }


def _print_summary(all_results, log_file):
    if not all_results:
        return

    print('\n' + '=' * 80)
    print('SUMMARY: All Datasets (with CI95 for each metric)')
    print('=' * 80)

    for res in all_results:
        ds = res.get('dataset', '?')
        print(f'\n--- {ds} ---')
        for key in ['Dice', 'HD95']:
            v = res.get(key, {})
            mean_v = v.get('mean', float('nan'))
            ci = v.get('CI95', (float('nan'), float('nan')))
            print(f'  {key}: mean={mean_v:.4f}, CI95=({ci[0]:.4f}, {ci[1]:.4f})')

        mal = res.get('Malignancy', {})
        if mal:
            print('  Malignancy:')
            for k, v in sorted(mal.items()):
                if isinstance(v, dict):
                    mean_v = v.get('mean', float('nan'))
                    ci = v.get('CI95', (float('nan'), float('nan')))
                    print(f'    {k}: mean={mean_v:.4f}, CI95=({ci[0]:.4f}, {ci[1]:.4f})')

        tir = res.get('TIRADS', {})
        if tir:
            print('  TIRADS:')
            for k, v in sorted(tir.items()):
                if isinstance(v, dict):
                    mean_v = v.get('mean', float('nan'))
                    ci = v.get('CI95', (float('nan'), float('nan')))
                    print(f'    {k}: mean={mean_v:.4f}, CI95=({ci[0]:.4f}, {ci[1]:.4f})')

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
        print(f'\nResults saved to: {json_path}')
    except Exception as e:
        print(f'Warning: Could not save JSON results: {e}')


def main():
    parser = argparse.ArgumentParser('DINOV3-UNet Test (Youden Threshold from Val)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_image_paths', type=str, action='append', default=[])
    parser.add_argument('--test_gt_paths', type=str, action='append', default=[])
    parser.add_argument('--test_label_paths', type=str, action='append', default=[])
    parser.add_argument('--test_dataset_names', type=str, action='append', default=[])
    parser.add_argument('--threshold_malignancy', type=float, default=None,
                        help='若设置则直接使用该阈值，不再用 Youden 计算；未设置时需提供 val_* 路径')
    parser.add_argument('--val_image_path', type=str, default=None)
    parser.add_argument('--val_gt_path', type=str, default=None)
    parser.add_argument('--val_label_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='./predictions')
    parser.add_argument('--save_results', type=str, default='true')
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='CUDA device index to use (default: 0)')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--img_size', type=int, default=224,
                        help='input image size for dataset (default: 224)')
    parser.add_argument('--dino_pretrained', type=str, default='True',
                        help='whether to load pretrained weights for the DINO backbone (True/False)')
    parser.add_argument('--use_dilation', type=str, default='False',
                        help='whether to use dilation layers in the model (True/False, default: False)')
    args = parser.parse_args()

    args.dino_pretrained = str(args.dino_pretrained).lower() in ('true', '1', 'yes', 'y')
    args.use_dilation = str(args.use_dilation).lower() in ('true', '1', 'yes', 'y')

    if args.threshold_malignancy is None:
        if not args.val_image_path or not args.val_gt_path or not args.val_label_path:
            parser.error('--val_image_path, --val_gt_path, --val_label_path are required when --threshold_malignancy is not set')

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_dir, f'test_youden_threshold_{timestamp}.log')
    sys.stdout = Logger(log_file)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda_device}')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')
    print(f'CUDA visible devices: {args.cuda_device}')
    print(f'Image size: {args.img_size} x {args.img_size}')
    print(f'DINO pretrained: {args.dino_pretrained}')
    print(f'DINO Dilation: {args.use_dilation}')
    print(f'Checkpoint path: {args.checkpoint}')
    print(f'Save path: {args.save_path}')
    print(f'Save results: {args.save_results}')
    print(f'Test dataset names: {args.test_dataset_names}')
    if args.threshold_malignancy is not None:
        print(f'Threshold malignancy (fixed): {args.threshold_malignancy}')
    else:
        print(f'Val image path: {args.val_image_path}')
        print(f'Val gt path: {args.val_gt_path}')
        print(f'Val label path: {args.val_label_path}')

    print('Loading model...')
    model = DINOv3_S_UNet_MULTITASK(pretrained=args.dino_pretrained, use_dilation=args.use_dilation).to(device)
    checkpoint_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint_dict, strict=False)
    print(f'Successfully loaded checkpoint from {args.checkpoint}\n')

    if args.threshold_malignancy is not None:
        best_threshold = float(args.threshold_malignancy)
        threshold_info = {
            'best_threshold': best_threshold,
            'youden': float('nan'),
            'sensitivity': float('nan'),
            'specificity': float('nan'),
        }
        print(f'Using fixed Malignancy threshold: {best_threshold:.4f} (--threshold_malignancy)\n')
    else:
        args.val_image_path = clean_path(args.val_image_path)
        args.val_gt_path = clean_path(args.val_gt_path)
        args.val_label_path = clean_path(args.val_label_path)
        print(f'Loading validation dataset with MultiTaskDataset using size: {args.img_size}x{args.img_size}')
        val_dataset = MultiTaskDataset(args.val_image_path, args.val_gt_path, args.val_label_path, args.img_size, mode='test')
        if len(val_dataset) == 0:
            print('Error: Validation dataset is empty!')
            return
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=12, num_workers=8)
        print(f'Validation dataset loaded: {len(val_dataset)} images found')
        print('Computing Malignancy threshold on validation set (Youden)...')
        threshold_info = compute_youden_threshold(model, val_loader, device, target_field='malignancy')
        best_threshold = float(threshold_info.get('best_threshold', 0.5))
        print(
            f"Youden Best Threshold (val): {threshold_info.get('best_threshold', float('nan'))}, "
            f"Youden: {threshold_info.get('youden', float('nan'))}, "
            f"Sensitivity: {threshold_info.get('sensitivity', float('nan'))}, "
            f"Specificity: {threshold_info.get('specificity', float('nan'))}\n"
        )

    args.test_image_paths = [clean_path(path) for path in args.test_image_paths]
    args.test_gt_paths = [clean_path(path) for path in args.test_gt_paths]
    if args.test_label_paths:
        args.test_label_paths = [clean_path(path) if path else None for path in args.test_label_paths]
    else:
        args.test_label_paths = [None] * len(args.test_image_paths)

    if len(args.test_image_paths) != len(args.test_gt_paths):
        print(
            f'Warning: Number of test image paths ({len(args.test_image_paths)}) and mask paths ({len(args.test_gt_paths)}) do not match.'
        )
        print('Using the minimum number of pairs.')
        min_len = min(len(args.test_image_paths), len(args.test_gt_paths))
        args.test_image_paths = args.test_image_paths[:min_len]
        args.test_gt_paths = args.test_gt_paths[:min_len]

    os.makedirs(args.save_path, exist_ok=True)
    print(f'Created base save directory: {args.save_path}')

    start_time = time.time()
    all_results = []
    for i, (img_path, gt_path) in enumerate(zip(args.test_image_paths, args.test_gt_paths)):
        if i < len(args.test_dataset_names) and args.test_dataset_names[i]:
            dataset_name = args.test_dataset_names[i]
        else:
            dataset_name = f'Test_Set_{i + 1}'

        label_path = args.test_label_paths[i] if i < len(args.test_label_paths) else None

        print(f'\nProcessing dataset {i + 1}/{len(args.test_image_paths)}')
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
    print(f'All datasets processed in {total_time:.2f} seconds')
    _print_summary(all_results, log_file)

    try:
        if hasattr(sys.stdout, 'close') and hasattr(sys.stdout, '_closed') and not sys.stdout._closed:
            sys.stdout.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
