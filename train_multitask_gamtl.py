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
from utils.metrics import evaluate_model
from utils.loss import structure_loss, benign_malignant_loss_gla, tirads_loss_gla
from utils.utils import gla_params, log_print

parser = argparse.ArgumentParser("DINOV3-UNet with GA-MTL (task-level gradient accumulation, no batch filtering)")
parser.add_argument("--method", type=str, required=True)
parser.add_argument("--train_image_path", type=str, required=True,
                    help="path to the images used for training")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the masks used for training")
parser.add_argument("--test_image_paths", type=str, nargs='+', required=True,
                    help="paths to the test image datasets")
parser.add_argument("--test_mask_paths", type=str, nargs='+', required=True,
                    help="paths to the test mask datasets")
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

# CUDA device index (use cuda:{index} when CUDA is available)
parser.add_argument('--cuda_device', type=int, default=0,
                    help='CUDA device index to use (default: 0)')

parser.add_argument('--use_dilation', type=str, default='False',
                    help='whether to use dilation layers in the model (True/False, default: False)')

parser.add_argument('--train_label_path', type=str, default=None,
                    help='path to the JSON file containing classification labels')
parser.add_argument('--test_label_paths', type=str, nargs='+', default=None,
                    help='paths to the test JSON files containing classification labels')
parser.add_argument('--gla_tau', type=float, default=None,
                    help='GLA tau: 1.0=full adjustment, 0.0=no adjustment. If None, auto based on imbalance')

# ===== GA-MTL knobs =====
parser.add_argument('--task_schedule', type=str, default='seg,bm',
                    help='Comma-separated task schedule per optimizer step. Default: seg,bm,tirads')
parser.add_argument('--steps_per_epoch', type=int, default=None,
                    help='Number of optimizer steps per epoch. If None, use len(dataloader)//len(schedule).')
parser.add_argument('--use_amp', action='store_true', help='Use torch.cuda.amp mixed precision')

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
    """Fetch next batch; restart iterator when exhausted."""
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


def _summarize_metrics(log_file, args, epoch_eval_points, epoch_dice_scores, epoch_hd95_scores, epoch_bm_metrics, epoch_tirads_metrics):
    log_print("\n========== Evaluation Summary (All Test Records) ==========\n", log_file)

    for dataset_name in args.test_dataset_names:
        eval_epochs = epoch_eval_points.get(dataset_name, [])
        log_print(f"\nDataset: {dataset_name}\n", log_file)
        log_print(f"Eval epochs: {eval_epochs}\n", log_file)

        # Segmentation
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

        # Benign/Malignant
        bm_list = epoch_bm_metrics.get(dataset_name, [])
        if bm_list:
            acc = [m.get('accuracy', 0.0) for m in bm_list]
            prec = [m.get('precision', 0.0) for m in bm_list]
            rec = [m.get('recall', 0.0) for m in bm_list]
            f1 = [m.get('f1_score', 0.0) for m in bm_list]
            auroc = [m.get('auroc', 0.0) for m in bm_list]
            auprc = [m.get('auprc', 0.0) for m in bm_list]
            ece = [m.get('ece', 0.0) for m in bm_list]

            log_print("\nBM metrics per eval:\n", log_file)
            log_print(f"  Acc:  {acc}\n", log_file)
            log_print(f"  Prec: {prec}\n", log_file)
            log_print(f"  Rec:  {rec}\n", log_file)
            log_print(f"  F1:   {f1}\n", log_file)
            log_print(f"  AUROC:{auroc}\n", log_file)
            log_print(f"  AUPRC:{auprc}\n", log_file)
            log_print(f"  ECE:  {ece}\n", log_file)

            bi = _best_index(f1, mode='max')
            log_print(f"Best BM F1: {f1[bi]:.4f} at epoch {eval_epochs[bi]}\n", log_file)
        else:
            log_print("\nBM metrics per eval: []\n", log_file)

        # TI-RADS
        ti_list = epoch_tirads_metrics.get(dataset_name, [])
        if ti_list:
            acc = [m.get('accuracy', 0.0) for m in ti_list]
            prec = [m.get('precision', 0.0) for m in ti_list]
            rec = [m.get('recall', 0.0) for m in ti_list]
            f1 = [m.get('f1_score', 0.0) for m in ti_list]
            auc = [m.get('auc', 0.0) for m in ti_list]
            ece = [m.get('ece', 0.0) for m in ti_list]

            log_print("\nTI-RADS metrics per eval:\n", log_file)
            log_print(f"  Acc:  {acc}\n", log_file)
            log_print(f"  Prec: {prec}\n", log_file)
            log_print(f"  Rec:  {rec}\n", log_file)
            log_print(f"  F1:   {f1}\n", log_file)
            log_print(f"  AUC:  {auc}\n", log_file)
            log_print(f"  ECE:  {ece}\n", log_file)

            bi = _best_index(f1, mode='max')
            log_print(f"Best TI-RADS F1: {f1[bi]:.4f} at epoch {eval_epochs[bi]}\n", log_file)
        else:
            log_print("\nTI-RADS metrics per eval: []\n", log_file)

    log_print("\n==========================================================\n", log_file)


def _new_metrics_to_legacy_tuple(results: dict):
    """Convert utils.new_metrics.evaluate_model() output to legacy (dice, hd95, bm_dict, tirads_dict).

    Legacy format is used by existing logging/tensorboard code.
    """
    dice = float(results.get('Dice', {}).get('mean', 0.0))
    hd95 = float(results.get('HD95', {}).get('mean', 0.0))

    malignancy = results.get('Malignancy', {}) or {}
    tirads = results.get('TIRADS', {}) or {}

    bm_m = {k: float(v.get('mean', 0.0)) for k, v in malignancy.items() if isinstance(v, dict)}
    ti_m = {k: float(v.get('mean', 0.0)) for k, v in tirads.items() if isinstance(v, dict)}

    # Fill expected keys if missing
    for k in ('accuracy', 'precision', 'recall', 'f1'):
        bm_m.setdefault(k, 0.0)
    bm_m.setdefault('f1_score', bm_m.get('f1', 0.0))
    bm_m.setdefault('auroc', 0.0)
    bm_m.setdefault('auprc', 0.0)
    bm_m.setdefault('ece', 0.0)

    for k in ('accuracy', 'precision', 'recall', 'f1', 'auc'):
        ti_m.setdefault(k, 0.0)
    ti_m.setdefault('f1_score', ti_m.get('f1', 0.0))
    ti_m.setdefault('ece', 0.0)

    return dice, hd95, bm_m, ti_m


def main(args):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join("logs", args.dataset_name, f'{args.method}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    log_name = os.path.join(save_dir, f'{args.method}_{args.dataset_name}_log.log')
    log_file = open(log_name, 'w')

    try:
        log_print(f"Training started at {time.ctime()}\n", log_file)
        log_print(f"Method: {args.method}\nDataset: {args.dataset_name}\nEpochs: {args.epoch}\n"
                  f"LR: {args.lr}\nBatch Size: {args.batch_size}\n", log_file)
        log_print(f"Image Size: {args.img_size} x {args.img_size}\n", log_file)
        log_print(f"DINO pretrained: {args.dino_pretrained}\n", log_file)
        log_print(f"Use Dilation: {args.use_dilation}\n", log_file)

        # ===== Data =====
        dataset = MultiTaskDataset(args.train_image_path, args.train_mask_path, args.train_label_path, args.img_size, mode='train')
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.cuda_device}")
        else:
            device = torch.device("cpu")

        # ===== Imbalance params (GLA) =====
        p_benign, p_malignant, p_tirads, gla_tau = gla_params(args.train_label_path, gla_tau=args.gla_tau, log_file=log_file)

        # ===== Model =====
        model = DINOv3_S_UNet_MULTITASK(pretrained=args.dino_pretrained, use_dilation=args.use_dilation)
        model.to(device)

        optim = opt.AdamW([{"params": model.parameters(), "initial_lr": args.lr}],
                          lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

        writer = tensorboardX.SummaryWriter(os.path.join(save_dir, 'tensorboard_logs'))
        scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp))

        # ===== Loss weights (keep same semantics as your original script) =====
        seg_weight = 1.0
        bm_weight = 1.0
        tirads_weight = 1.0
        log_print(f"Loss weights: seg={seg_weight}, bm={bm_weight}, tirads={tirads_weight}\n", log_file)

        # ===== GA-MTL schedule =====
        schedule = [t.strip() for t in args.task_schedule.split(',') if t.strip()]
        assert len(schedule) > 0, "Empty task_schedule"
        for t in schedule:
            assert t in ('seg', 'bm', 'tirads'), f"Unknown task in schedule: {t}"
        log_print(f"GA-MTL schedule per optimizer step: {schedule}\n", log_file)

        # One optimizer step accumulates gradients from len(schedule) task batches.
        if args.steps_per_epoch is None:
            steps_per_epoch = max(1, len(dataloader) // len(schedule))
        else:
            steps_per_epoch = args.steps_per_epoch
        log_print(f"steps_per_epoch (optimizer steps): {steps_per_epoch}\n", log_file)

        # ===== Metrics buffers =====
        epoch_dice_scores = {name: [] for name in args.test_dataset_names}
        epoch_hd95_scores = {name: [] for name in args.test_dataset_names}
        epoch_bm_metrics = {name: [] for name in args.test_dataset_names}
        epoch_tirads_metrics = {name: [] for name in args.test_dataset_names}
        epoch_eval_points = {name: [] for name in args.test_dataset_names}

        if args.test_label_paths is None:
            args.test_label_paths = [None] * len(args.test_dataset_names)

        it = _reset_iter(dataloader)
        global_step = 0

        for epoch in range(args.epoch):
            model.train()

            # epoch average stats (task-wise)
            seg_loss_sum = 0.0
            bm_loss_sum = 0.0
            ti_loss_sum = 0.0
            n_seg = 0
            n_bm = 0
            n_ti = 0

            for step in range(steps_per_epoch):
                optim.zero_grad(set_to_none=True)

                # === Task-level gradient accumulation:
                # For each task in schedule:
                #   - take one batch (no task-availability filtering)
                #   - compute task-specific loss
                #   - backward (accumulate gradients)
                # After all tasks:
                #   - optimizer.step() once
                for task_name in schedule:
                    batch, it = _next_batch(it, dataloader)

                    x = batch['image'].to(device)
                    with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
                        pred_seg, pred_bm, pred_ti = model(x)

                        if task_name == 'seg':
                            target = batch['label'].to(device)
                            loss = seg_weight * structure_loss(pred_seg, target)
                            seg_loss_sum += float(loss.detach().item())
                            n_seg += 1

                        elif task_name == 'bm':
                            # Assumption (as you stated): ALL samples have valid BM labels.
                            y = batch['malignancy'].to(device)
                            loss = bm_weight * benign_malignant_loss_gla(
                                pred_bm, y,
                                p_pos=p_malignant, p_neg=p_benign, tau=gla_tau
                            )
                            bm_loss_sum += float(loss.detach().item())
                            n_bm += 1

                        elif task_name == 'tirads':
                            # Only a subset of samples have valid TIRADS labels (-1 means missing).
                            # We DO NOT filter batches; we just ignore invalid labels inside the loss computation.
                            y = batch['tirads'].to(device)
                            valid = (y != -1)
                            if valid.any():
                                loss = tirads_weight * tirads_loss_gla(
                                    pred_ti[valid], y[valid],
                                    p_class=p_tirads, tau=gla_tau, label_smoothing=0.0
                                )
                                ti_loss_sum += float(loss.detach().item())
                                n_ti += 1
                            else:
                                # No valid labels in this batch -> no gradient contribution for this task.
                                # Keep it as a zero loss to preserve "task identity" in the schedule.
                                loss = pred_ti.sum() * 0.0

                        else:
                            raise ValueError(f"Unknown task: {task_name}")

                    # Normalize by number of scheduled tasks to keep gradient scale stable.
                    loss = loss / float(len(schedule))
                    scaler.scale(loss).backward()

                scaler.step(optim)
                scaler.update()

                global_step += 1
                writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step)

            scheduler.step()

            # ===== epoch logs =====
            seg_loss_avg = seg_loss_sum / max(1, n_seg)
            bm_loss_avg = bm_loss_sum / max(1, n_bm)
            ti_loss_avg = ti_loss_sum / max(1, n_ti) if n_ti > 0 else 0.0

            log_print(
                f"Epoch:{epoch+1}: seg_loss(avg):{seg_loss_avg:.4f}, "
                f"bm_loss(avg):{bm_loss_avg:.4f}, tirads_loss(avg):{ti_loss_avg:.4f}\n",
                log_file
            )

            # ===== checkpoints =====
            checkpoint_path = os.path.join(args.dir_checkpoint, args.dataset_name, timestamp)
            os.makedirs(checkpoint_path, exist_ok=True)
            if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.epoch - 1:
                ckpt = os.path.join(checkpoint_path, f"{args.method}_{args.dataset_name}_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), str(ckpt))
                log_print(f"Saved checkpoint at epoch {epoch+1}\n", log_file)

            # ===== evaluation =====
            if (epoch + 1) % args.eval_interval == 0:
                log_print(f"\nValidating epoch {epoch+1} on all test datasets...\n", log_file)
                model.eval()
                with torch.no_grad():
                    for dataset_name, img_path, mask_path, test_label_path in zip(
                        args.test_dataset_names, args.test_image_paths, args.test_mask_paths, args.test_label_paths
                    ):
                        epoch_eval_points[dataset_name].append(epoch + 1)
                        log_print(f"Testing on {dataset_name}...\n", log_file)
                        test_dataset = MultiTaskDataset(img_path, mask_path, test_label_path, args.img_size, mode='test')
                        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

                        # new_metrics.evaluate_model returns a dict (mean + CI95). Convert to legacy tuple.
                        eval_results = evaluate_model(model, test_loader, device, threshold=0.5)
                        dice, hd95, bm_m, ti_m = _new_metrics_to_legacy_tuple(eval_results)

                        epoch_dice_scores[dataset_name].append(dice)
                        epoch_hd95_scores[dataset_name].append(hd95)
                        epoch_bm_metrics[dataset_name].append(bm_m)
                        epoch_tirads_metrics[dataset_name].append(ti_m)

                        writer.add_scalar(f'VAL_DICE/{dataset_name}', dice, epoch+1)
                        writer.add_scalar(f'VAL_HD95/{dataset_name}', hd95, epoch+1)

                        writer.add_scalar(f'VAL_BM_ACC/{dataset_name}', bm_m.get('accuracy', 0.0), epoch+1)
                        writer.add_scalar(f'VAL_BM_F1/{dataset_name}', bm_m.get('f1_score', 0.0), epoch+1)
                        writer.add_scalar(f'VAL_BM_AUROC/{dataset_name}', bm_m.get('auroc', 0.0), epoch+1)
                        writer.add_scalar(f'VAL_BM_AUPRC/{dataset_name}', bm_m.get('auprc', 0.0), epoch+1)
                        writer.add_scalar(f'VAL_BM_ECE/{dataset_name}', bm_m.get('ece', 0.0), epoch+1)

                        writer.add_scalar(f'VAL_TIRADS_ACC/{dataset_name}', ti_m.get('accuracy', 0.0), epoch+1)
                        writer.add_scalar(f'VAL_TIRADS_F1/{dataset_name}', ti_m.get('f1_score', 0.0), epoch+1)
                        writer.add_scalar(f'VAL_TIRADS_AUC/{dataset_name}', ti_m.get('auc', 0.0), epoch+1)
                        writer.add_scalar(f'VAL_TIRADS_ECE/{dataset_name}', ti_m.get('ece', 0.0), epoch+1)

                        log_print(f"  {dataset_name} - Dice: {dice:.4f}, HD95: {hd95:.4f}\n", log_file)
                        log_print(
                            f"  {dataset_name} - BM: Acc={bm_m['accuracy']:.4f}, "
                            f"Prec={bm_m['precision']:.4f}, Rec={bm_m['recall']:.4f}, "
                            f"F1={bm_m['f1_score']:.4f}, AUROC={bm_m.get('auroc', 0.0):.4f}, "
                            f"AUPRC={bm_m.get('auprc', 0.0):.4f}, ECE={bm_m.get('ece', 0.0):.4f}\n",
                            log_file
                        )
                        log_print(
                            f"  {dataset_name} - TI-RADS: Acc={ti_m['accuracy']:.4f}, "
                            f"Prec={ti_m['precision']:.4f}, Rec={ti_m['recall']:.4f}, "
                            f"F1={ti_m['f1_score']:.4f}, AUC={ti_m.get('auc', 0.0):.4f}, "
                            f"ECE={ti_m.get('ece', 0.0):.4f}\n",
                            log_file
                        )

        _summarize_metrics(
            log_file,
            args,
            epoch_eval_points,
            epoch_dice_scores,
            epoch_hd95_scores,
            epoch_bm_metrics,
            epoch_tirads_metrics,
        )

        log_print(f"\n========== Training Completed ==========\n", log_file)
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
