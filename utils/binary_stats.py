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
