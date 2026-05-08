import json
from collections import defaultdict

import numpy as np

from utils.utils import log_print


def _auto_gla_tau(p_neg, p_pos, gla_tau, log_file=None):
    imbalance_ratio = max(p_neg, p_pos) / min(p_neg, p_pos)

    if gla_tau is not None:
        log_print(f"Using manual tau={gla_tau:.2f}\n", log_file)
        return gla_tau

    if imbalance_ratio < 1.2:
        gla_tau = 0.3
    elif imbalance_ratio < 1.5:
        gla_tau = 0.5
    elif imbalance_ratio < 2.5:
        gla_tau = 0.7
    else:
        gla_tau = 1.0

    log_print(f"Auto-adjusted tau={gla_tau:.2f} (imbalance_ratio={imbalance_ratio:.2f})\n", log_file)
    return gla_tau


def _load_label_items(label_json_path):
    with open(label_json_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    if isinstance(label_data, dict):
        return list(label_data.values())
    return label_data


def _derive_patient_id(filename: str, depth: int = 2) -> str:
    normalized = filename.replace('\\', '/').strip('/')
    parts = [part for part in normalized.split('/') if part]
    if not parts:
        raise ValueError('filename is empty after normalization')
    if depth <= 0:
        raise ValueError(f'depth must be positive, got {depth}')
    return '/'.join(parts[:min(depth, len(parts))])


def _build_patient_target_map(items, target_key, patient_id_depth):
    patient_targets = defaultdict(set)
    patient_missing = set()

    for item in items:
        filename = item.get('filename')
        if not filename:
            continue
        patient_id = _derive_patient_id(filename, patient_id_depth)
        target = item.get(target_key, -1)

        if target in (0, 1):
            patient_targets[patient_id].add(int(target))
        else:
            patient_missing.add(patient_id)

    patient_target_map = {}
    for patient_id, target_values in patient_targets.items():
        if len(target_values) > 1:
            raise ValueError(
                f"Inconsistent targets found for patient_id={patient_id}, target_key={target_key}: {sorted(target_values)}"
            )
        patient_target_map[patient_id] = next(iter(target_values))

    for patient_id in patient_missing:
        patient_target_map.setdefault(patient_id, -1)

    return patient_target_map


def summarize_binary_label_distribution(label_json_path, target_key, log_file=None):
    items = _load_label_items(label_json_path)

    neg_count = sum(1 for item in items if item.get(target_key) == 0)
    pos_count = sum(1 for item in items if item.get(target_key) == 1)
    missing_count = sum(1 for item in items if item.get(target_key, -1) == -1)
    total_binary = neg_count + pos_count

    epsilon = 1e-8
    if total_binary > 0:
        p_neg = (neg_count + epsilon) / (total_binary + 2 * epsilon)
        p_pos = (pos_count + epsilon) / (total_binary + 2 * epsilon)
    else:
        p_neg = 0.5
        p_pos = 0.5

    log_print(
        f"Training set - {target_key}: Negative={neg_count} ({p_neg:.2%}), Positive={pos_count} ({p_pos:.2%}), Missing={missing_count}\n",
        log_file,
    )

    return {
        'negative_count': neg_count,
        'positive_count': pos_count,
        'missing_count': missing_count,
        'valid_count': total_binary,
        'p_neg': p_neg,
        'p_pos': p_pos,
    }


def summarize_binary_label_distribution_patient(label_json_path, target_key, patient_id_depth=2, log_file=None):
    items = _load_label_items(label_json_path)
    patient_target_map = _build_patient_target_map(items, target_key, patient_id_depth)

    neg_count = sum(1 for target in patient_target_map.values() if target == 0)
    pos_count = sum(1 for target in patient_target_map.values() if target == 1)
    missing_count = sum(1 for target in patient_target_map.values() if target == -1)
    total_binary = neg_count + pos_count

    epsilon = 1e-8
    if total_binary > 0:
        p_neg = (neg_count + epsilon) / (total_binary + 2 * epsilon)
        p_pos = (pos_count + epsilon) / (total_binary + 2 * epsilon)
    else:
        p_neg = 0.5
        p_pos = 0.5

    log_print(
        (
            f"Training set - {target_key} (patient-level, patient_id_depth={patient_id_depth}): "
            f"Negative={neg_count} ({p_neg:.2%}), Positive={pos_count} ({p_pos:.2%}), Missing={missing_count}\n"
        ),
        log_file,
    )

    return {
        'negative_count': neg_count,
        'positive_count': pos_count,
        'missing_count': missing_count,
        'valid_count': total_binary,
        'patient_count': len(patient_target_map),
        'p_neg': p_neg,
        'p_pos': p_pos,
    }


def gla_params_binary(train_label_path, target_key, gla_tau, log_file=None):
    if train_label_path:
        stats = summarize_binary_label_distribution(train_label_path, target_key, log_file=log_file)
        p_neg = stats['p_neg']
        p_pos = stats['p_pos']
        gla_tau = _auto_gla_tau(p_neg, p_pos, gla_tau, log_file=log_file)
        log_print(f"GLA parameters for {target_key}: p_neg={p_neg:.4f}, p_pos={p_pos:.4f}, tau={gla_tau:.2f}\n", log_file)
        log_print(f"Logit adjustment magnitude: {abs(gla_tau * (np.log(p_pos) - np.log(p_neg))):.4f}\n", log_file)
    else:
        p_neg = 0.5
        p_pos = 0.5
        gla_tau = gla_tau if gla_tau is not None else 0.5

    return p_neg, p_pos, gla_tau


def gla_params_binary_patient(train_label_path, target_key, patient_id_depth=2, gla_tau=None, log_file=None):
    if train_label_path:
        stats = summarize_binary_label_distribution_patient(
            train_label_path,
            target_key,
            patient_id_depth=patient_id_depth,
            log_file=log_file,
        )
        p_neg = stats['p_neg']
        p_pos = stats['p_pos']
        gla_tau = _auto_gla_tau(p_neg, p_pos, gla_tau, log_file=log_file)
        log_print(
            f"Patient-level GLA parameters for {target_key}: p_neg={p_neg:.4f}, p_pos={p_pos:.4f}, tau={gla_tau:.2f}\n",
            log_file,
        )
        log_print(f"Logit adjustment magnitude: {abs(gla_tau * (np.log(p_pos) - np.log(p_neg))):.4f}\n", log_file)
    else:
        p_neg = 0.5
        p_pos = 0.5
        gla_tau = gla_tau if gla_tau is not None else 0.5

    return p_neg, p_pos, gla_tau


def compute_binary_pos_weight_patient(label_json_path, target_key, patient_id_depth=2, log_file=None):
    stats = summarize_binary_label_distribution_patient(
        label_json_path,
        target_key,
        patient_id_depth=patient_id_depth,
        log_file=log_file,
    )
    neg_count = stats['negative_count']
    pos_count = stats['positive_count']

    if neg_count == 0 or pos_count == 0:
        raise ValueError(
            f"Cannot compute patient-level pos_weight for {target_key}: negative_count={neg_count}, positive_count={pos_count}. Both classes must exist."
        )

    pos_weight = float(neg_count) / float(pos_count)
    log_print(f"Patient-level pos_weight for {target_key}: {pos_weight:.6f}\n", log_file)
    return pos_weight


def build_binary_sampler_weights(samples, target_key, pos_fraction=0.3, log_file=None):
    if not (0.0 < float(pos_fraction) < 1.0):
        raise ValueError(f"pos_fraction must be in (0, 1), got {pos_fraction}")

    neg_count = 0
    pos_count = 0
    missing_count = 0

    for sample in samples:
        target = sample.get('target', -1)
        if target == 0:
            neg_count += 1
        elif target == 1:
            pos_count += 1
        else:
            missing_count += 1

    if neg_count == 0 or pos_count == 0:
        raise ValueError(
            f"Cannot build binary sampler weights for {target_key}: "
            f"negative_count={neg_count}, positive_count={pos_count}. Both classes must exist."
        )

    valid_count = neg_count + pos_count
    neg_fraction = 1.0 - float(pos_fraction)
    pos_weight = float(pos_fraction) / float(pos_count)
    neg_weight = neg_fraction / float(neg_count)
    missing_weight = 0.0

    sample_weights = []
    for sample in samples:
        target = sample.get('target', -1)
        if target == 0:
            sample_weights.append(neg_weight)
        elif target == 1:
            sample_weights.append(pos_weight)
        else:
            sample_weights.append(missing_weight)

    expected_pos_draws = valid_count * float(pos_fraction)
    expected_neg_draws = valid_count * neg_fraction

    log_print(
        f"Sampler stats for {target_key}: Negative={neg_count}, Positive={pos_count}, Missing={missing_count}, Valid={valid_count}\n",
        log_file,
    )
    log_print(
        f"Sampler config for {target_key}: target_pos_fraction={float(pos_fraction):.4f}, target_neg_fraction={neg_fraction:.4f}\n",
        log_file,
    )
    log_print(
        f"Sampler weights for {target_key}: neg_weight={neg_weight:.8f}, pos_weight={pos_weight:.8f}, missing_weight={missing_weight:.8f}\n",
        log_file,
    )
    log_print(
        f"Expected sampled draws per default epoch for {target_key}: Negative≈{expected_neg_draws:.1f}, Positive≈{expected_pos_draws:.1f}\n",
        log_file,
    )
    log_print(
        f"Expected repeat factor per default epoch for {target_key}: neg≈{expected_neg_draws / neg_count:.2f}x, pos≈{expected_pos_draws / pos_count:.2f}x\n",
        log_file,
    )

    return {
        'negative_count': neg_count,
        'positive_count': pos_count,
        'missing_count': missing_count,
        'valid_count': valid_count,
        'target_pos_fraction': float(pos_fraction),
        'target_neg_fraction': neg_fraction,
        'neg_weight': neg_weight,
        'pos_weight': pos_weight,
        'missing_weight': missing_weight,
        'sample_weights': sample_weights,
    }
