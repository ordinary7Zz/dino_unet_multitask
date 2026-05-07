import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import random
import numpy as np
import time

import torch
import torch.optim as opt
import tensorboardX
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import MultiTaskDataset
from dino_unet_multitask import DINOv3_S_UNet_MULTITASK
from utils.metrics import evaluate_model_binary_target
from utils.loss import structure_loss, benign_malignant_loss
from utils.utils import log_print

parser = argparse.ArgumentParser("DINOV3-UNet multitask binary classification")
parser.add_argument("--method", type=str, required=True)
parser.add_argument("--train_image_path", type=str, required=True,
                    help="path to the training image root")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the training mask root")
parser.add_argument("--test_image_paths", type=str, nargs='+', required=True,
                    help="paths to the test image roots")
parser.add_argument("--test_mask_paths", type=str, nargs='+', required=True,
                    help="paths to the test mask roots")
parser.add_argument("--test_dataset_names", type=str, nargs='+', required=True,
                    help="names of the test datasets")

parser.add_argument("--epoch", type=int, default=50, help="training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
parser.add_argument("--batch_size", default=12, type=int)

parser.add_argument('--dir_checkpoint', type=str, default='/checkpoint/')
parser.add_argument('--checkpoint_interval', type=int, default=1)
parser.add_argument('--eval_interval', type=int, default=5)
parser.add_argument('--dataset_name', type=str, default='default')
parser.add_argument('--img_size', type=int, default=224,
                    help='input image size for dataset (default: 224)')
parser.add_argument('--dino_pretrained', type=str, default='True',
                    help='whether to load pretrained weights for the DINO backbone (True/False)')
parser.add_argument('--cuda_device', type=int, default=0,
                    help='CUDA device index to use (default: 0)')
parser.add_argument('--use_dilation', type=str, default='False',
                    help='whether to use dilation layers in the model (True/False, default: False)')

parser.add_argument('--train_label_path', type=str, required=True,
                    help='path to the JSON file containing training labels')
parser.add_argument('--test_label_paths', type=str, nargs='+', required=True,
                    help='paths to the test JSON files containing classification labels')
parser.add_argument('--target_key', type=str, required=True,
                    help='binary target key in the label JSON, e.g. LNM_CN01 or FTCPTC')

parser.add_argument('--task_schedule', type=str, default='seg,cls',
                    help='Comma-separated task schedule per optimizer step. Default: seg,cls')
parser.add_argument('--steps_per_epoch', type=int, default=None,
                    help='Number of optimizer steps per epoch. If None, use len(dataloader)//len(schedule).')
parser.add_argument('--use_amp', action='store_true', help='Use torch.cuda.amp mixed precision')
parser.add_argument('--cls_pos_weight', type=float, default=None,
                    help='Positive class weight for binary BCE loss. Larger values emphasize recall for positive class.')

args = parser.parse_args()
args.dino_pretrained = str(args.dino_pretrained).lower() in ('true', '1', 'yes', 'y')
args.use_dilation = str(args.use_dilation).lower() in ('true', '1', 'yes', 'y')


def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _reset_iter(dataloader):
    return iter(dataloader)


def _next_batch(it, dataloader):
    try:
        return next(it), it
    except StopIteration:
        it = _reset_iter(dataloader)
        return next(it), it


def _best_index(values, mode: str):
    if not values:
        return None
    if mode == 'max':
        return int(np.argmax(values))
    if mode == 'min':
        return int(np.argmin(values))
    raise ValueError(f"Unknown mode: {mode}")


def _binary_metrics_to_flat_dict(metrics_dict):
    flat = {k: float(v.get('mean', 0.0)) for k, v in metrics_dict.items() if isinstance(v, dict)}
    for key in ('accuracy', 'precision', 'recall', 'f1', 'auroc', 'auprc', 'sensitivity', 'specificity', 'youden', 'ece'):
        flat.setdefault(key, 0.0)
    flat['f1_score'] = flat.get('f1', 0.0)
    return flat


def _summarize_metrics(log_file, args, epoch_eval_points, epoch_dice_scores, epoch_hd95_scores, epoch_cls_metrics):
    log_print("\n========== Evaluation Summary (All Test Records) ==========\n\n", log_file)

    for dataset_name in args.test_dataset_names:
        eval_epochs = epoch_eval_points.get(dataset_name, [])
        log_print(f"\nDataset: {dataset_name}\n", log_file)
        log_print(f"Eval epochs: {eval_epochs}\n", log_file)

        dice_list = epoch_dice_scores.get(dataset_name, [])
        hd95_list = epoch_hd95_scores.get(dataset_name, [])
        log_print(f"Dice per eval: {dice_list}\n", log_file)
        log_print(f"HD95 per eval: {hd95_list}\n", log_file)

        if dice_list:
            bi = _best_index(dice_list, mode='max')
            log_print(f"Best Dice: {dice_list[bi]:.4f} at epoch {eval_epochs[bi]}\n", log_file)
        if hd95_list:
            bi = _best_index(hd95_list, mode='min')
            log_print(f"Best HD95: {hd95_list[bi]:.4f} at epoch {eval_epochs[bi]}\n", log_file)

        cls_list = epoch_cls_metrics.get(dataset_name, [])
        if cls_list:
            acc = [m.get('accuracy', 0.0) for m in cls_list]
            prec = [m.get('precision', 0.0) for m in cls_list]
            rec = [m.get('recall', 0.0) for m in cls_list]
            f1 = [m.get('f1_score', 0.0) for m in cls_list]
            auroc = [m.get('auroc', 0.0) for m in cls_list]
            auprc = [m.get('auprc', 0.0) for m in cls_list]
            ece = [m.get('ece', 0.0) for m in cls_list]

            log_print(f"\n{args.target_key} metrics per eval:\n", log_file)
            log_print(f"  Acc:  {acc}\n", log_file)
            log_print(f"  Prec: {prec}\n", log_file)
            log_print(f"  Rec:  {rec}\n", log_file)
            log_print(f"  F1:   {f1}\n", log_file)
            log_print(f"  AUROC:{auroc}\n", log_file)
            log_print(f"  AUPRC:{auprc}\n", log_file)
            log_print(f"  ECE:  {ece}\n", log_file)

            bi = _best_index(f1, mode='max')
            log_print(f"Best {args.target_key} F1: {f1[bi]:.4f} at epoch {eval_epochs[bi]}\n", log_file)
        else:
            log_print(f"\n{args.target_key} metrics per eval: []\n", log_file)

    log_print("\n==========================================================\n", log_file)


def main(args):
    if not (len(args.test_image_paths) == len(args.test_mask_paths) == len(args.test_dataset_names) == len(args.test_label_paths)):
        raise ValueError('test_image_paths, test_mask_paths, test_dataset_names, test_label_paths must have the same length')

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('logs', args.dataset_name, f'{args.method}_{args.target_key}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    log_name = os.path.join(save_dir, f'{args.method}_{args.dataset_name}_{args.target_key}_log.log')
    log_file = open(log_name, 'w', encoding='utf-8')

    try:
        log_print(f"Training started at {time.ctime()}\n", log_file)
        log_print(f"Method: {args.method}\nDataset: {args.dataset_name}\nEpochs: {args.epoch}\n", log_file)
        log_print(f"Target Key: {args.target_key}\nLR: {args.lr}\nBatch Size: {args.batch_size}\n", log_file)
        log_print(f"Image Size: {args.img_size} x {args.img_size}\n", log_file)
        log_print(f"DINO pretrained: {args.dino_pretrained}\n", log_file)
        log_print(f"Use Dilation: {args.use_dilation}\n", log_file)

        dataset = MultiTaskDataset(
            args.train_image_path,
            args.train_mask_path,
            args.train_label_path,
            args.img_size,
            mode='train',
            target_key=args.target_key,
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        log_print(
            f"Using DataLoader shuffle=True, dataset_size={len(dataset)}, batch_size={args.batch_size}, drop_last=True\n",
            log_file,
        )
        first_sample = dataset.samples[0]
        log_print(
            f"First sample: filename={first_sample['filename']}, image={first_sample['image_path']}, mask={first_sample['mask_path']}, target={first_sample['target']}\n",
            log_file
        )

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.cuda_device}")
        else:
            device = torch.device('cpu')

        model = DINOv3_S_UNet_MULTITASK(pretrained=args.dino_pretrained, use_dilation=args.use_dilation)
        model.to(device)

        optim = opt.AdamW([{"params": model.parameters(), "initial_lr": args.lr}],
                          lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

        writer = tensorboardX.SummaryWriter(os.path.join(save_dir, 'tensorboard_logs'))
        scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp))

        seg_weight = 1.0
        cls_weight = 1.0
        log_print(f"Loss weights: seg={seg_weight}, cls={cls_weight}\n", log_file)
        log_print(f"Classification pos_weight: {args.cls_pos_weight}\n", log_file)

        schedule = [t.strip() for t in args.task_schedule.split(',') if t.strip()]
        assert len(schedule) > 0, 'Empty task_schedule'
        for t in schedule:
            assert t in ('seg', 'cls'), f"Unknown task in schedule: {t}"
        log_print(f"GA-MTL schedule per optimizer step: {schedule}\n", log_file)

        if args.steps_per_epoch is None:
            steps_per_epoch = max(1, len(dataloader) // len(schedule))
        else:
            steps_per_epoch = args.steps_per_epoch
        log_print(f"steps_per_epoch (optimizer steps): {steps_per_epoch}\n", log_file)

        epoch_dice_scores = {name: [] for name in args.test_dataset_names}
        epoch_hd95_scores = {name: [] for name in args.test_dataset_names}
        epoch_cls_metrics = {name: [] for name in args.test_dataset_names}
        epoch_eval_points = {name: [] for name in args.test_dataset_names}

        it = _reset_iter(dataloader)
        global_step = 0

        for epoch in range(args.epoch):
            model.train()

            seg_loss_sum = 0.0
            cls_loss_sum = 0.0
            n_seg = 0
            n_cls = 0

            for step in range(steps_per_epoch):
                optim.zero_grad(set_to_none=True)

                for task_name in schedule:
                    batch, it = _next_batch(it, dataloader)

                    x = batch['image'].to(device)
                    with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
                        pred_seg, pred_cls, _ = model(x)

                        if task_name == 'seg':
                            target = batch['label'].to(device)
                            loss = seg_weight * structure_loss(pred_seg, target)
                            seg_loss_sum += float(loss.detach().item())
                            n_seg += 1
                        elif task_name == 'cls':
                            y = batch['target'].to(device)
                            valid = (y != -1)
                            if valid.any():
                                loss = cls_weight * benign_malignant_loss(
                                    pred_cls[valid],
                                    y[valid],
                                    pos_weight=args.cls_pos_weight,
                                )
                                cls_loss_sum += float(loss.detach().item())
                                n_cls += 1
                            else:
                                loss = pred_cls.sum() * 0.0
                        else:
                            raise ValueError(f"Unknown task: {task_name}")

                    loss = loss / float(len(schedule))
                    scaler.scale(loss).backward()

                scaler.step(optim)
                scaler.update()

                global_step += 1
                writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step)

            scheduler.step()

            seg_loss_avg = seg_loss_sum / max(1, n_seg)
            cls_loss_avg = cls_loss_sum / max(1, n_cls) if n_cls > 0 else 0.0
            log_print(
                f"Epoch:{epoch + 1}: seg_loss(avg):{seg_loss_avg:.4f}, cls_loss(avg):{cls_loss_avg:.4f}\n",
                log_file
            )
            writer.add_scalar('TRAIN_SEG_LOSS', seg_loss_avg, epoch + 1)
            writer.add_scalar('TRAIN_CLS_LOSS', cls_loss_avg, epoch + 1)

            checkpoint_path = os.path.join(args.dir_checkpoint, args.dataset_name, timestamp)
            os.makedirs(checkpoint_path, exist_ok=True)
            if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.epoch - 1:
                ckpt = os.path.join(checkpoint_path, f"{args.method}_{args.dataset_name}_{args.target_key}_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), str(ckpt))
                log_print(f"Saved checkpoint at epoch {epoch + 1}\n", log_file)

            if (epoch + 1) % args.eval_interval == 0:
                log_print(f"\nValidating epoch {epoch + 1} on all test datasets...\n", log_file)
                model.eval()
                with torch.no_grad():
                    for dataset_name, img_path, mask_path, test_label_path in zip(
                        args.test_dataset_names, args.test_image_paths, args.test_mask_paths, args.test_label_paths
                    ):
                        epoch_eval_points[dataset_name].append(epoch + 1)
                        log_print(f"Testing on {dataset_name}...\n", log_file)
                        test_dataset = MultiTaskDataset(
                            img_path,
                            mask_path,
                            test_label_path,
                            args.img_size,
                            mode='test',
                            target_key=args.target_key,
                        )
                        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

                        eval_results = evaluate_model_binary_target(
                            model,
                            test_loader,
                            device,
                            threshold=0.5,
                            target_field='target',
                            target_name=args.target_key,
                        )
                        dice = float(eval_results.get('Dice', {}).get('mean', 0.0))
                        hd95 = float(eval_results.get('HD95', {}).get('mean', 0.0))
                        cls_metrics = _binary_metrics_to_flat_dict(eval_results.get(args.target_key, {}))

                        epoch_dice_scores[dataset_name].append(dice)
                        epoch_hd95_scores[dataset_name].append(hd95)
                        epoch_cls_metrics[dataset_name].append(cls_metrics)

                        writer.add_scalar(f'VAL_DICE/{dataset_name}', dice, epoch + 1)
                        writer.add_scalar(f'VAL_HD95/{dataset_name}', hd95, epoch + 1)
                        writer.add_scalar(f'VAL_CLS_ACC/{dataset_name}', cls_metrics.get('accuracy', 0.0), epoch + 1)
                        writer.add_scalar(f'VAL_CLS_F1/{dataset_name}', cls_metrics.get('f1_score', 0.0), epoch + 1)
                        writer.add_scalar(f'VAL_CLS_AUROC/{dataset_name}', cls_metrics.get('auroc', 0.0), epoch + 1)
                        writer.add_scalar(f'VAL_CLS_AUPRC/{dataset_name}', cls_metrics.get('auprc', 0.0), epoch + 1)
                        writer.add_scalar(f'VAL_CLS_ECE/{dataset_name}', cls_metrics.get('ece', 0.0), epoch + 1)

                        log_print(f"  {dataset_name} - Dice: {dice:.4f}, HD95: {hd95:.4f}\n", log_file)
                        log_print(
                            f"  {dataset_name} - {args.target_key}: Acc={cls_metrics['accuracy']:.4f}, "
                            f"Prec={cls_metrics['precision']:.4f}, Rec={cls_metrics['recall']:.4f}, "
                            f"F1={cls_metrics['f1_score']:.4f}, AUROC={cls_metrics['auroc']:.4f}, "
                            f"AUPRC={cls_metrics['auprc']:.4f}, ECE={cls_metrics['ece']:.4f}\n",
                            log_file
                        )

        _summarize_metrics(
            log_file,
            args,
            epoch_eval_points,
            epoch_dice_scores,
            epoch_hd95_scores,
            epoch_cls_metrics,
        )

        log_print("\n========== Training Completed ==========\n", log_file)
        log_print(f"Training completed at {time.ctime()}\n", log_file)

    except Exception as e:
        import traceback
        error_msg = f"Error occurred: {str(e)}\n{traceback.format_exc()}\n"
        log_print(error_msg, log_file)
        raise
    finally:
        log_file.close()
        if 'writer' in locals():
            writer.close()


if __name__ == "__main__":
    seed_torch(1024)
    main(args)
