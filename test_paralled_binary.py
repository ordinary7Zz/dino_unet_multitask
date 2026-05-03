import argparse
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from dataset import MultiTaskDataset
from dino_unet_multitask import DINOv3_S_UNet_MULTITASK
from utils.binary_test_utils import clean_path, process_binary_dataset, print_binary_summary
from utils.metrics import compute_youden_threshold


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


def main():
    parser = argparse.ArgumentParser('DINOV3-UNet Binary Test (Target-Key Aware)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--target_key', type=str, required=True,
                        help='binary target key in the label JSON, e.g. LNM_CN01 or FTCPTC')
    parser.add_argument('--test_image_paths', type=str, action='append', default=[])
    parser.add_argument('--test_gt_paths', type=str, action='append', default=[])
    parser.add_argument('--test_label_paths', type=str, action='append', default=[])
    parser.add_argument('--test_dataset_names', type=str, action='append', default=[])
    parser.add_argument('--threshold_malignancy', type=float, default=None,
                        help='fixed decision threshold for the selected binary target')
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
    log_file = os.path.join(args.log_dir, f'test_binary_{args.target_key}_{timestamp}.log')
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
    print(f'Target key: {args.target_key}')
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
        print(f'Using fixed {args.target_key} threshold: {best_threshold:.4f} (--threshold_malignancy)\n')
    else:
        args.val_image_path = clean_path(args.val_image_path)
        args.val_gt_path = clean_path(args.val_gt_path)
        args.val_label_path = clean_path(args.val_label_path)
        print(f'Loading validation dataset with MultiTaskDataset using size: {args.img_size}x{args.img_size}')
        val_dataset = MultiTaskDataset(
            args.val_image_path,
            args.val_gt_path,
            args.val_label_path,
            args.img_size,
            mode='test',
            target_key=args.target_key,
        )
        if len(val_dataset) == 0:
            print('Error: Validation dataset is empty!')
            return
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=12, num_workers=8)
        print(f'Validation dataset loaded: {len(val_dataset)} images found')
        print(f'Computing {args.target_key} threshold on validation set (Youden)...')
        threshold_info = compute_youden_threshold(
            model,
            val_loader,
            device,
            target_field='target',
        )
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

    all_results = []
    for i, (img_path, gt_path) in enumerate(zip(args.test_image_paths, args.test_gt_paths)):
        if i < len(args.test_dataset_names) and args.test_dataset_names[i]:
            dataset_name = args.test_dataset_names[i]
        else:
            dataset_name = f'Test_Set_{i + 1}'

        label_path = args.test_label_paths[i] if i < len(args.test_label_paths) else None

        print(f'\nProcessing dataset {i + 1}/{len(args.test_image_paths)}')
        dataset_result = process_binary_dataset(
            model,
            args.img_size,
            threshold_info,
            img_path,
            gt_path,
            label_path,
            args.save_path,
            dataset_name,
            args.target_key,
            device,
            args.save_results,
            threshold=best_threshold,
            use_fixed_threshold=(args.threshold_malignancy is not None),
        )
        all_results.append(dataset_result)

    print_binary_summary(all_results, log_file, target_key=args.target_key)

    try:
        if hasattr(sys.stdout, 'close') and hasattr(sys.stdout, '_closed') and not sys.stdout._closed:
            sys.stdout.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
