from typing import Optional

import torch
from tqdm import tqdm

from utils.metrics import (
    Dice,
    HD95,
    _segmentation_results_dict,
    classification_bootstrap_metrics,
    find_best_threshold_by_youden_index,
)


def pool_instance_logits(instance_logits: torch.Tensor, bag_sizes, pooling: str = 'max') -> torch.Tensor:
    if instance_logits.dim() == 2 and instance_logits.size(1) == 1:
        logits = instance_logits.squeeze(1)
    elif instance_logits.dim() == 1:
        logits = instance_logits
    else:
        raise ValueError(f'Expected binary logits shaped [N] or [N, 1], got {tuple(instance_logits.shape)}')

    if isinstance(bag_sizes, torch.Tensor):
        bag_sizes = bag_sizes.tolist()

    pooled_logits = []
    start = 0
    for bag_size in bag_sizes:
        if bag_size <= 0:
            raise ValueError(f'bag_size must be positive, got {bag_size}')
        end = start + int(bag_size)
        bag_logits = logits[start:end]
        if pooling == 'max':
            pooled_logits.append(torch.max(bag_logits))
        elif pooling == 'mean':
            pooled_logits.append(torch.mean(bag_logits))
        else:
            raise ValueError(f'Unsupported pooling mode: {pooling}')
        start = end

    if start != logits.size(0):
        raise ValueError(f'bag_sizes sum to {start}, but logits contain {logits.size(0)} instances')

    return torch.stack(pooled_logits, dim=0)


def collect_patient_predictions(net, dataloader, device, cls_pooling: str = 'max'):
    was_training = net.training
    net.eval()

    patient_ids = []
    all_target_probs = []
    all_target_labels = []

    for batch in tqdm(dataloader, desc='Collecting Patient Predictions', leave=False):
        image = batch['image'].to(device)
        target_labels = batch['target'].to(device)
        bag_sizes = batch['bag_sizes']

        with torch.no_grad():
            outputs = net(image)
            if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
                raise ValueError('Model outputs must include binary classification logits at index 1')
            target_logits = outputs[1]
            pooled_logits = pool_instance_logits(target_logits, bag_sizes, pooling=cls_pooling)
            target_probs = torch.sigmoid(pooled_logits)

        valid = (target_labels != -1)
        if valid.any():
            valid_idx = torch.where(valid)[0].tolist()
            all_target_probs.extend(target_probs[valid].detach().cpu().numpy().tolist())
            all_target_labels.extend(target_labels[valid].detach().cpu().numpy().tolist())
            patient_ids.extend([batch['patient_ids'][i] for i in valid_idx])

    if was_training:
        net.train()

    return {
        'patient_ids': patient_ids,
        'probs': all_target_probs,
        'labels': all_target_labels,
    }


def compute_patient_youden_threshold(
    net,
    dataloader,
    device,
    thresholds: Optional[torch.Tensor] = None,
    cls_pooling: str = 'max',
):
    prediction_dict = collect_patient_predictions(net, dataloader, device, cls_pooling=cls_pooling)
    return find_best_threshold_by_youden_index(
        prediction_dict['labels'],
        prediction_dict['probs'],
        thresholds=thresholds,
    )


def evaluate_model_binary_target_patient(
    net,
    dataloader,
    device,
    threshold: float = 0.5,
    target_name: str = 'BinaryTarget',
    cls_pooling: str = 'max',
):
    net.eval()

    dice_calculator = Dice()
    hd_calculator = HD95()

    all_dice_values = []
    all_hd_values = []
    all_target_probs = []
    all_target_labels = []

    for batch in tqdm(dataloader, desc='Evaluating Patient Model', leave=False):
        image = batch['image'].to(device)
        mask_true = batch['label'].to(device)
        target_labels = batch['target'].to(device)
        bag_sizes = batch['bag_sizes']

        with torch.no_grad():
            outputs = net(image)
            if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
                raise ValueError('Model outputs must include segmentation logits and binary logits')

            mask_pred = outputs[0]
            target_logits = outputs[1]
            pooled_logits = pool_instance_logits(target_logits, bag_sizes, pooling=cls_pooling)
            target_probs = torch.sigmoid(pooled_logits)

            valid = (target_labels != -1)
            if valid.any():
                all_target_probs.extend(target_probs[valid].detach().cpu().numpy().tolist())
                all_target_labels.extend(target_labels[valid].detach().cpu().numpy().tolist())

            mask_pred = torch.sigmoid(mask_pred)
            mask_pred_binary = (mask_pred > 0.5).float()

        batch_size = image.size(0)
        for i in range(batch_size):
            pred_i = mask_pred_binary[i]
            true_i = (mask_true[i] > 0.5).float()
            all_dice_values.append(dice_calculator(pred_i, true_i).item())
            try:
                all_hd_values.append(hd_calculator(pred_i, true_i).item())
            except Exception as exc:
                print(f'[Warning] HD95 failed on sample {i}: {exc}')

    net.train()

    results = _segmentation_results_dict(all_dice_values, all_hd_values)
    results[target_name] = {}

    binary_metrics_ci = classification_bootstrap_metrics(all_target_probs, all_target_labels, threshold=threshold)
    for key, value in binary_metrics_ci.items():
        mean_v, (low_v, high_v) = value
        results[target_name][key] = {
            'mean': round(mean_v, 4),
            'CI95': (round(low_v, 4), round(high_v, 4)),
        }

    return results
