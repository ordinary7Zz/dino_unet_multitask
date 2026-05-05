import json
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


def build_binary_sampler_weights(samples, target_key, log_file=None):
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

    neg_weight = 1.0 / float(neg_count)
    pos_weight = 1.0 / float(pos_count)
    missing_weight = min(neg_weight, pos_weight)

    sample_weights = []
    for sample in samples:
        target = sample.get('target', -1)
        if target == 0:
            sample_weights.append(neg_weight)
        elif target == 1:
            sample_weights.append(pos_weight)
        else:
            sample_weights.append(missing_weight)

    log_print(
        f"Sampler stats for {target_key}: Negative={neg_count}, Positive={pos_count}, Missing={missing_count}\n",
        log_file,
    )
    log_print(
        f"Sampler weights for {target_key}: neg_weight={neg_weight:.8f}, pos_weight={pos_weight:.8f}, missing_weight={missing_weight:.8f}\n",
        log_file,
    )

    return {
        'negative_count': neg_count,
        'positive_count': pos_count,
        'missing_count': missing_count,
        'neg_weight': neg_weight,
        'pos_weight': pos_weight,
        'missing_weight': missing_weight,
        'sample_weights': sample_weights,
    }
