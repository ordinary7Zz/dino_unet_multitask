from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from build_patient_filtered_labels import (
        compute_bbox_stats,
        compute_mask_stats,
        derive_patient_id,
        load_image_and_mask,
        load_items,
        normalize_relative_path,
        resolve_mask_path,
        save_json,
        score_item,
        summarize_image_level,
        summarize_patient_level,
    )
    from analyze_patient_heterogeneity import (
        DEFAULT_HETEROGENEITY_FEATURES,
        compute_patient_feature_summary,
        compute_patient_heterogeneity,
        detect_within_patient_outliers,
        extract_image_features,
    )
except ImportError:
    from utils.build_patient_filtered_labels import (
        compute_bbox_stats,
        compute_mask_stats,
        derive_patient_id,
        load_image_and_mask,
        load_items,
        normalize_relative_path,
        resolve_mask_path,
        save_json,
        score_item,
        summarize_image_level,
        summarize_patient_level,
    )
    from utils.analyze_patient_heterogeneity import (
        DEFAULT_HETEROGENEITY_FEATURES,
        compute_patient_feature_summary,
        compute_patient_heterogeneity,
        detect_within_patient_outliers,
        extract_image_features,
    )


TASK_PRESETS = {
    'FTCPTC': {
        'min_mask_area_ratio': 0.0005,
        'min_largest_component_pixels': 64,
        'min_keep_per_patient': 4,
        'max_keep_per_patient': 6,
        'min_images_per_patient': 2,
        'outlier_z_threshold': 2.5,
        'outlier_mad_threshold': 3.5,
        'high_heterogeneity_percentile': 90.0,
        'review_drop_ratio': 0.8,
        'heterogeneity_features': list(DEFAULT_HETEROGENEITY_FEATURES),
        'score_weights': {
            'largest_component_ratio_reward': 0.75,
            'fragmentation_penalty': 0.85,
            'occupancy_irregularity_penalty': 0.45,
            'edge_touch_penalty': 0.60,
            'center_distance_penalty': 0.35,
            'outlier_penalty': 0.20,
        },
    },
    'LNM_CN01': {
        'min_mask_area_ratio': 0.0008,
        'min_largest_component_pixels': 96,
        'min_keep_per_patient': 2,
        'max_keep_per_patient': 4,
        'min_images_per_patient': 2,
        'outlier_z_threshold': 2.5,
        'outlier_mad_threshold': 3.5,
        'high_heterogeneity_percentile': 90.0,
        'review_drop_ratio': 0.7,
        'heterogeneity_features': list(DEFAULT_HETEROGENEITY_FEATURES),
        'score_weights': {
            'largest_component_ratio_reward': 0.55,
            'fragmentation_penalty': 1.15,
            'occupancy_irregularity_penalty': 0.80,
            'edge_touch_penalty': 1.05,
            'center_distance_penalty': 0.60,
            'outlier_penalty': 0.35,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build task-suitable image-level labels by combining hard filtering with heterogeneity-aware ranking.'
    )
    parser.add_argument('--input_json', type=str, required=True, help='Path to the original label JSON file')
    parser.add_argument('--image_root', type=str, required=True, help='Path to the image root directory')
    parser.add_argument('--mask_root', type=str, required=True, help='Path to the mask root directory')
    parser.add_argument('--output_json', type=str, required=True, help='Path to save the filtered label JSON file')
    parser.add_argument('--target_key', type=str, required=True, help='Binary target key to filter, e.g. FTCPTC or LNM_CN01')
    parser.add_argument('--task_preset', type=str, default=None,
                        help='Optional task preset. Defaults to target_key when it matches a built-in preset')
    parser.add_argument('--patient_id_depth', type=int, default=2, help='Number of path segments used to derive patient_id')
    parser.add_argument('--min_mask_area_ratio', type=float, default=None,
                        help='Override the preset minimum foreground area ratio')
    parser.add_argument('--min_largest_component_pixels', type=int, default=None,
                        help='Override the preset minimum largest connected-component size in pixels')
    parser.add_argument('--min_keep_per_patient', type=int, default=None,
                        help='Override the preset minimum number of kept images per patient')
    parser.add_argument('--max_keep_per_patient', type=int, default=None,
                        help='Override the preset maximum number of kept images per patient')
    parser.add_argument('--min_images_per_patient', type=int, default=None,
                        help='Override the minimum images per patient needed for heterogeneity scoring')
    parser.add_argument('--outlier_z_threshold', type=float, default=None,
                        help='Override the z-score threshold used when MAD is unavailable')
    parser.add_argument('--outlier_mad_threshold', type=float, default=None,
                        help='Override the modified-z threshold used when MAD is available')
    parser.add_argument('--high_heterogeneity_percentile', type=float, default=None,
                        help='Override the percentile cutoff used to flag high-heterogeneity patients')
    parser.add_argument('--review_drop_ratio', type=float, default=None,
                        help='Override the patient review threshold based on dropped ratio')
    parser.add_argument('--heterogeneity_features', type=str, default=None,
                        help='Comma-separated feature list; defaults to the preset feature set')
    parser.add_argument('--report_dir', type=str, default=None, help='Directory to save summary and audit reports')
    parser.add_argument('--dry_run', action='store_true', help='Analyze and write reports without saving filtered label JSON')
    return parser.parse_args()


def get_task_preset(task_name: str):
    normalized = task_name.strip().upper()
    if normalized not in TASK_PRESETS:
        raise ValueError(
            f'Unsupported task preset: {task_name}. Available presets: {", ".join(sorted(TASK_PRESETS))}'
        )
    preset = TASK_PRESETS[normalized]
    return {
        'name': normalized,
        'min_mask_area_ratio': float(preset['min_mask_area_ratio']),
        'min_largest_component_pixels': int(preset['min_largest_component_pixels']),
        'min_keep_per_patient': int(preset['min_keep_per_patient']),
        'max_keep_per_patient': int(preset['max_keep_per_patient']),
        'min_images_per_patient': int(preset['min_images_per_patient']),
        'outlier_z_threshold': float(preset['outlier_z_threshold']),
        'outlier_mad_threshold': float(preset['outlier_mad_threshold']),
        'high_heterogeneity_percentile': float(preset['high_heterogeneity_percentile']),
        'review_drop_ratio': float(preset['review_drop_ratio']),
        'heterogeneity_features': list(preset['heterogeneity_features']),
        'score_weights': dict(preset['score_weights']),
    }


def parse_feature_list(raw_value: str | None, default_features):
    if raw_value is None:
        return list(default_features)
    features = [part.strip() for part in raw_value.split(',') if part.strip()]
    if not features:
        raise ValueError('heterogeneity_features is empty after parsing')
    return features


def resolve_config(args):
    preset_name = args.task_preset or args.target_key
    config = get_task_preset(preset_name)
    config['target_key'] = args.target_key
    config['patient_id_depth'] = args.patient_id_depth
    config['min_mask_area_ratio'] = float(args.min_mask_area_ratio) if args.min_mask_area_ratio is not None else config['min_mask_area_ratio']
    config['min_largest_component_pixels'] = (
        int(args.min_largest_component_pixels)
        if args.min_largest_component_pixels is not None
        else config['min_largest_component_pixels']
    )
    config['min_keep_per_patient'] = int(args.min_keep_per_patient) if args.min_keep_per_patient is not None else config['min_keep_per_patient']
    config['max_keep_per_patient'] = int(args.max_keep_per_patient) if args.max_keep_per_patient is not None else config['max_keep_per_patient']
    config['min_images_per_patient'] = int(args.min_images_per_patient) if args.min_images_per_patient is not None else config['min_images_per_patient']
    config['outlier_z_threshold'] = float(args.outlier_z_threshold) if args.outlier_z_threshold is not None else config['outlier_z_threshold']
    config['outlier_mad_threshold'] = float(args.outlier_mad_threshold) if args.outlier_mad_threshold is not None else config['outlier_mad_threshold']
    config['high_heterogeneity_percentile'] = (
        float(args.high_heterogeneity_percentile)
        if args.high_heterogeneity_percentile is not None
        else config['high_heterogeneity_percentile']
    )
    config['review_drop_ratio'] = float(args.review_drop_ratio) if args.review_drop_ratio is not None else config['review_drop_ratio']
    config['heterogeneity_features'] = parse_feature_list(args.heterogeneity_features, config['heterogeneity_features'])

    if config['patient_id_depth'] <= 0:
        raise ValueError(f'patient_id_depth must be positive, got {config["patient_id_depth"]}')
    if config['min_mask_area_ratio'] < 0.0:
        raise ValueError(f'min_mask_area_ratio must be non-negative, got {config["min_mask_area_ratio"]}')
    if config['min_largest_component_pixels'] < 0:
        raise ValueError(
            f'min_largest_component_pixels must be non-negative, got {config["min_largest_component_pixels"]}'
        )
    if config['min_keep_per_patient'] < 0:
        raise ValueError(f'min_keep_per_patient must be non-negative, got {config["min_keep_per_patient"]}')
    if config['max_keep_per_patient'] is not None and config['max_keep_per_patient'] <= 0:
        raise ValueError(f'max_keep_per_patient must be positive when provided, got {config["max_keep_per_patient"]}')
    if config['max_keep_per_patient'] is not None and config['max_keep_per_patient'] < config['min_keep_per_patient']:
        raise ValueError('max_keep_per_patient must be greater than or equal to min_keep_per_patient')
    if config['min_images_per_patient'] <= 0:
        raise ValueError(f'min_images_per_patient must be positive, got {config["min_images_per_patient"]}')
    if not (0.0 <= config['review_drop_ratio'] <= 1.0):
        raise ValueError(f'review_drop_ratio must be in [0, 1], got {config["review_drop_ratio"]}')
    if not (0.0 <= config['high_heterogeneity_percentile'] <= 100.0):
        raise ValueError(
            f'high_heterogeneity_percentile must be in [0, 100], got {config["high_heterogeneity_percentile"]}'
        )

    return config


def build_structural_drop_decision(index: int, item, filename, patient_id, target, reason: str, details: dict):
    return {
        'index': index,
        'item': item,
        'filename': filename,
        'patient_id': patient_id,
        'target': target,
        'hard_filter_status': 'drop',
        'hard_drop_reason': reason,
        'hard_drop_details': details,
        'feature_values': None,
        'outlier_flags': [],
        'outlier_score': None,
        'base_score': None,
        'composite_score': None,
        'rank_within_patient': None,
        'final_decision': 'drop',
        'final_drop_reason': reason,
    }


def build_analyzed_items(items, image_root: Path, mask_root: Path, patient_id_depth: int, target_key: str):
    analyzed_items = []
    structural_decisions = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            structural_decisions.append(
                build_structural_drop_decision(
                    index=index,
                    item=item,
                    filename=None,
                    patient_id=None,
                    target=None,
                    reason='invalid_item_type',
                    details={'type': type(item).__name__},
                )
            )
            continue

        filename = item.get('filename')
        if not filename:
            structural_decisions.append(
                build_structural_drop_decision(
                    index=index,
                    item=item,
                    filename=None,
                    patient_id=None,
                    target=item.get(target_key, -1),
                    reason='missing_filename',
                    details={},
                )
            )
            continue

        normalized_filename = normalize_relative_path(filename)
        patient_id = derive_patient_id(filename, patient_id_depth)
        image_path = image_root / Path(normalized_filename)
        mask_path = resolve_mask_path(mask_root, normalized_filename)
        raw_target = item.get(target_key, -1)

        if not image_path.exists():
            structural_decisions.append(
                build_structural_drop_decision(
                    index=index,
                    item=item,
                    filename=filename,
                    patient_id=patient_id,
                    target=raw_target,
                    reason='missing_image_file',
                    details={'image_path': str(image_path)},
                )
            )
            continue

        if mask_path is None or not mask_path.exists():
            structural_decisions.append(
                build_structural_drop_decision(
                    index=index,
                    item=item,
                    filename=filename,
                    patient_id=patient_id,
                    target=raw_target,
                    reason='missing_mask_file',
                    details={'mask_root': str(mask_root)},
                )
            )
            continue

        try:
            image_array, mask_array = load_image_and_mask(image_path, mask_path)
        except Exception as exc:
            structural_decisions.append(
                build_structural_drop_decision(
                    index=index,
                    item=item,
                    filename=filename,
                    patient_id=patient_id,
                    target=raw_target,
                    reason='load_error',
                    details={'error': str(exc)},
                )
            )
            continue

        mask_stats = compute_mask_stats(mask_array)
        bbox_stats = compute_bbox_stats(mask_stats)
        features = extract_image_features(image_array, mask_array, mask_stats, bbox_stats)
        analyzed_items.append({
            'index': index,
            'item': item,
            'filename': filename,
            'normalized_filename': normalized_filename,
            'patient_id': patient_id,
            'target': raw_target,
            'image_path': str(image_path),
            'mask_path': str(mask_path),
            'mask_stats': mask_stats,
            'bbox_stats': bbox_stats,
            'features': features,
            'base_score': float(score_item(mask_stats, bbox_stats)),
            'decision': {
                'index': index,
                'item': item,
                'filename': filename,
                'patient_id': patient_id,
                'target': raw_target,
                'hard_filter_status': 'pending',
                'hard_drop_reason': None,
                'hard_drop_details': None,
                'feature_values': features,
                'outlier_flags': [],
                'outlier_score': None,
                'base_score': float(score_item(mask_stats, bbox_stats)),
                'composite_score': None,
                'rank_within_patient': None,
                'final_decision': None,
                'final_drop_reason': None,
            },
        })

    return analyzed_items, structural_decisions


def apply_hard_filter(entry, reason: str, details: dict):
    decision = entry['decision']
    decision['hard_filter_status'] = 'drop'
    decision['hard_drop_reason'] = reason
    decision['hard_drop_details'] = details
    decision['final_decision'] = 'drop'
    decision['final_drop_reason'] = reason


def hard_filter_items(analyzed_items, config: dict):
    kept_candidates = []
    dropped_decisions = []

    for entry in analyzed_items:
        target = entry['target']
        mask_stats = entry['mask_stats']
        decision = entry['decision']

        if target not in (0, 1):
            apply_hard_filter(entry, 'invalid_target', {'target_value': target})
            dropped_decisions.append(decision)
            continue
        if mask_stats['foreground_pixels'] == 0:
            apply_hard_filter(entry, 'empty_mask', mask_stats)
            dropped_decisions.append(decision)
            continue
        if mask_stats['mask_area_ratio'] < config['min_mask_area_ratio']:
            apply_hard_filter(
                entry,
                'tiny_mask_area_ratio',
                {
                    'mask_area_ratio': mask_stats['mask_area_ratio'],
                    'threshold': config['min_mask_area_ratio'],
                    'foreground_pixels': mask_stats['foreground_pixels'],
                    'image_area': mask_stats['image_area'],
                },
            )
            dropped_decisions.append(decision)
            continue
        if mask_stats['largest_component_pixels'] < config['min_largest_component_pixels']:
            apply_hard_filter(
                entry,
                'tiny_largest_component',
                {
                    'largest_component_pixels': mask_stats['largest_component_pixels'],
                    'threshold': config['min_largest_component_pixels'],
                },
            )
            dropped_decisions.append(decision)
            continue

        decision['hard_filter_status'] = 'pass'
        kept_candidates.append(entry)

    return kept_candidates, dropped_decisions


def build_patient_records(candidate_entries, target_key: str):
    patient_groups = defaultdict(list)
    for entry in candidate_entries:
        patient_groups[entry['patient_id']].append(entry)

    patient_records = []
    patient_review_records = []
    dropped_decisions = []

    for patient_id in sorted(patient_groups.keys()):
        group = sorted(patient_groups[patient_id], key=lambda item: (item['filename'], item['index']))
        deduped_entries = []
        duplicate_filenames = []
        seen_filenames = set()

        for entry in group:
            if entry['filename'] in seen_filenames:
                duplicate_filenames.append(entry['filename'])
                apply_hard_filter(
                    entry,
                    'duplicate_filename',
                    {'filename': entry['filename'], 'patient_id': patient_id},
                )
                dropped_decisions.append(entry['decision'])
                continue
            seen_filenames.add(entry['filename'])
            deduped_entries.append(entry)

        if not deduped_entries:
            patient_review_records.append({
                'patient_id': patient_id,
                'reasons': ['all_images_dropped'],
                'label_values': [],
                'valid_image_count': 0,
                'kept_image_count': 0,
                'dropped_image_count': len(group),
                'duplicate_filenames': sorted(set(duplicate_filenames)),
            })
            continue

        label_values = sorted({int(entry['target']) for entry in deduped_entries if entry['target'] in (0, 1)})
        if len(label_values) > 1:
            for entry in deduped_entries:
                apply_hard_filter(
                    entry,
                    'inconsistent_patient_labels',
                    {'label_values': label_values, 'target_key': target_key},
                )
                dropped_decisions.append(entry['decision'])
            patient_review_records.append({
                'patient_id': patient_id,
                'reasons': ['inconsistent_patient_labels'],
                'label_values': label_values,
                'valid_image_count': len(deduped_entries),
                'kept_image_count': 0,
                'dropped_image_count': len(group),
                'duplicate_filenames': sorted(set(duplicate_filenames)),
            })
            continue

        patient_records.append({
            'patient_id': patient_id,
            'entries': deduped_entries,
            'target': label_values[0] if len(label_values) == 1 else None,
            'label_values': label_values,
            'duplicate_filenames': sorted(set(duplicate_filenames)),
        })

    return patient_records, patient_review_records, dropped_decisions


def build_outlier_map(outlier_records):
    outlier_map = {}
    for record in outlier_records:
        outlier_map[(record['patient_id'], record['filename'])] = record
    return outlier_map


def compute_center_distance(entry):
    center_x_ratio = entry['features'].get('center_x_ratio')
    center_y_ratio = entry['features'].get('center_y_ratio')
    if center_x_ratio is None or center_y_ratio is None:
        return 0.0
    return float(abs(center_x_ratio - 0.5) + abs(center_y_ratio - 0.5))


def compute_outlier_penalty(outlier_record, config: dict):
    if not outlier_record:
        return 0.0
    denominator = max(config['outlier_mad_threshold'], config['outlier_z_threshold'], 1.0)
    normalized = float(outlier_record['outlier_score']) / float(denominator)
    return float(min(normalized, 4.0))


def compute_composite_score(entry, outlier_record, config: dict):
    weights = config['score_weights']
    features = entry['features']
    largest_component_ratio = float(features.get('largest_component_ratio') or 0.0)
    component_fragmentation = float(features.get('component_fragmentation') or 0.0)
    occupancy_irregularity = float(features.get('occupancy_irregularity') or 0.0)
    edge_touch_penalty = float(features.get('edge_touch_penalty') or 0.0)
    center_distance = compute_center_distance(entry)
    outlier_penalty = compute_outlier_penalty(outlier_record, config)

    return float(
        entry['base_score']
        + weights['largest_component_ratio_reward'] * largest_component_ratio
        - weights['fragmentation_penalty'] * component_fragmentation
        - weights['occupancy_irregularity_penalty'] * occupancy_irregularity
        - weights['edge_touch_penalty'] * edge_touch_penalty
        - weights['center_distance_penalty'] * center_distance
        - weights['outlier_penalty'] * outlier_penalty
    )


def determine_keep_count(entry_count: int, min_keep_per_patient: int, max_keep_per_patient: int | None):
    if entry_count <= 0:
        return 0
    keep_count = entry_count if max_keep_per_patient is None else min(entry_count, max_keep_per_patient)
    if entry_count <= min_keep_per_patient:
        return entry_count
    return max(min_keep_per_patient, keep_count)


def summarize_bag_sizes(items, patient_id_depth: int, target_key: str):
    patient_counts = defaultdict(int)
    for item in items:
        if not isinstance(item, dict):
            continue
        filename = item.get('filename')
        if not filename:
            continue
        target = item.get(target_key, -1)
        if target not in (0, 1):
            continue
        patient_counts[derive_patient_id(filename, patient_id_depth)] += 1

    if not patient_counts:
        return {
            'patient_count': 0,
            'min': None,
            'median': None,
            'mean': None,
            'p90': None,
            'max': None,
        }

    values = np.asarray(sorted(patient_counts.values()), dtype=np.float64)
    return {
        'patient_count': int(values.size),
        'min': int(values.min()),
        'median': float(np.median(values)),
        'mean': float(values.mean()),
        'p90': float(np.percentile(values, 90)),
        'max': int(values.max()),
    }


def compute_high_heterogeneity_threshold(patient_results, percentile: float):
    scores = [result['heterogeneity_score'] for result in patient_results if result['heterogeneity_score'] is not None]
    if not scores:
        return None
    return float(np.percentile(np.asarray(scores, dtype=np.float64), percentile))


def select_topk_per_patient(patient_records, config: dict):
    patient_results = []
    outlier_records = []

    for patient_record in patient_records:
        feature_summary = compute_patient_feature_summary(patient_record['entries'], config['heterogeneity_features'])
        heterogeneity_score, per_feature_dispersion, heterogeneity_error = compute_patient_heterogeneity(
            patient_record['entries'],
            config['heterogeneity_features'],
            config['min_images_per_patient'],
        )
        patient_outliers = detect_within_patient_outliers(
            patient_record,
            config['heterogeneity_features'],
            config['outlier_z_threshold'],
            config['outlier_mad_threshold'],
        )
        outlier_records.extend(patient_outliers)
        outlier_map = build_outlier_map(patient_outliers)

        for entry in patient_record['entries']:
            outlier_record = outlier_map.get((patient_record['patient_id'], entry['filename']))
            entry['decision']['outlier_flags'] = list(outlier_record['triggered_features']) if outlier_record else []
            entry['decision']['outlier_score'] = float(outlier_record['outlier_score']) if outlier_record else None
            entry['decision']['composite_score'] = compute_composite_score(entry, outlier_record, config)

        ranked_entries = sorted(
            patient_record['entries'],
            key=lambda entry: (-entry['decision']['composite_score'], entry['filename'], entry['index']),
        )
        keep_count = determine_keep_count(
            len(ranked_entries),
            config['min_keep_per_patient'],
            config['max_keep_per_patient'],
        )
        kept_entries = ranked_entries[:keep_count]
        dropped_entries = ranked_entries[keep_count:]

        for rank, entry in enumerate(ranked_entries, start=1):
            decision = entry['decision']
            decision['rank_within_patient'] = rank
            if rank <= keep_count:
                decision['final_decision'] = 'floor_keep' if rank <= min(config['min_keep_per_patient'], keep_count) else 'topk_keep'
                decision['final_drop_reason'] = None
            else:
                decision['final_decision'] = 'drop'
                decision['final_drop_reason'] = 'deprioritized_topk'

        patient_results.append({
            'patient_id': patient_record['patient_id'],
            'label': patient_record['target'],
            'valid_image_count': len(ranked_entries),
            'kept_image_count': len(kept_entries),
            'dropped_image_count': len(dropped_entries),
            'kept_filenames': [entry['filename'] for entry in kept_entries],
            'dropped_filenames': [entry['filename'] for entry in dropped_entries],
            'outlier_count': len(patient_outliers),
            'outlier_filenames': sorted({record['filename'] for record in patient_outliers}),
            'duplicate_filenames': patient_record['duplicate_filenames'],
            'label_values': patient_record['label_values'],
            'min_keep_activated': len(ranked_entries) >= config['min_keep_per_patient'] and keep_count >= config['min_keep_per_patient'],
            'heterogeneity_score': heterogeneity_score,
            'heterogeneity_error': heterogeneity_error,
            'feature_summary': feature_summary,
            'per_feature_dispersion': per_feature_dispersion,
        })

    return patient_results, outlier_records


def build_reports(input_items, output_items, structural_decisions, hard_filter_drops, patient_stage_drops,
                  image_decisions, patient_results, patient_review_records, outlier_records, config: dict):
    high_threshold = compute_high_heterogeneity_threshold(
        patient_results,
        config['high_heterogeneity_percentile'],
    )

    patient_report = []
    review_records = list(patient_review_records)
    high_heterogeneity_patient_count = 0
    analyzable_patient_count = 0
    below_min_images_patient_count = 0

    for result in sorted(patient_results, key=lambda item: (
        item['heterogeneity_score'] is None,
        -(item['heterogeneity_score'] or -1.0),
        item['patient_id'],
    )):
        high_flag = (
            high_threshold is not None
            and result['heterogeneity_score'] is not None
            and result['heterogeneity_score'] >= high_threshold
        )
        if result['heterogeneity_score'] is not None:
            analyzable_patient_count += 1
        if result['heterogeneity_error'] == 'below_min_images':
            below_min_images_patient_count += 1
        if high_flag:
            high_heterogeneity_patient_count += 1

        drop_ratio = (
            float(result['dropped_image_count']) / float(result['valid_image_count'])
            if result['valid_image_count'] > 0 else 1.0
        )
        review_reasons = []
        if result['kept_image_count'] == 0:
            review_reasons.append('all_images_dropped')
        if drop_ratio >= config['review_drop_ratio']:
            review_reasons.append('high_drop_ratio')
        if result['heterogeneity_error'] == 'below_min_images':
            review_reasons.append('below_min_images')
        if high_flag:
            review_reasons.append('high_heterogeneity')
        if result['duplicate_filenames']:
            review_reasons.append('duplicate_filename')

        if review_reasons:
            review_records.append({
                'patient_id': result['patient_id'],
                'reasons': sorted(set(review_reasons)),
                'label_values': result['label_values'],
                'valid_image_count': result['valid_image_count'],
                'kept_image_count': result['kept_image_count'],
                'dropped_image_count': result['dropped_image_count'],
                'duplicate_filenames': result['duplicate_filenames'],
                'heterogeneity_score': result['heterogeneity_score'],
            })

        patient_report.append({
            'patient_id': result['patient_id'],
            'label': result['label'],
            'heterogeneity_score': result['heterogeneity_score'],
            'high_heterogeneity_flag': high_flag,
            'valid_image_count': result['valid_image_count'],
            'kept_image_count': result['kept_image_count'],
            'dropped_image_count': result['dropped_image_count'],
            'outlier_count': result['outlier_count'],
            'outlier_filenames': result['outlier_filenames'],
            'kept_filenames': result['kept_filenames'],
            'dropped_filenames': result['dropped_filenames'],
            'min_keep_activated': result['min_keep_activated'],
            'duplicate_filenames': result['duplicate_filenames'],
            'label_values': result['label_values'],
            'heterogeneity_error': result['heterogeneity_error'],
            'per_feature_dispersion': result['per_feature_dispersion'],
            'feature_summary': result['feature_summary'],
        })

    review_records.sort(key=lambda item: item['patient_id'])
    image_decisions_sorted = sorted(image_decisions, key=lambda item: (item['index'], item['filename'] or ''))
    outlier_records_sorted = sorted(outlier_records, key=lambda item: (-item['outlier_score'], item['patient_id'], item['filename']))

    summary = {
        'target_key': config['target_key'],
        'task_preset': config['name'],
        'config': {
            'patient_id_depth': config['patient_id_depth'],
            'min_mask_area_ratio': config['min_mask_area_ratio'],
            'min_largest_component_pixels': config['min_largest_component_pixels'],
            'min_keep_per_patient': config['min_keep_per_patient'],
            'max_keep_per_patient': config['max_keep_per_patient'],
            'min_images_per_patient': config['min_images_per_patient'],
            'outlier_z_threshold': config['outlier_z_threshold'],
            'outlier_mad_threshold': config['outlier_mad_threshold'],
            'high_heterogeneity_percentile': config['high_heterogeneity_percentile'],
            'review_drop_ratio': config['review_drop_ratio'],
            'heterogeneity_features': list(config['heterogeneity_features']),
            'score_weights': dict(config['score_weights']),
        },
        'input_image_level': summarize_image_level(input_items, config['target_key']),
        'output_image_level': summarize_image_level(output_items, config['target_key']),
        'input_patient_level': summarize_patient_level(input_items, config['patient_id_depth'], config['target_key']),
        'output_patient_level': summarize_patient_level(output_items, config['patient_id_depth'], config['target_key']),
        'input_bag_size_distribution': summarize_bag_sizes(input_items, config['patient_id_depth'], config['target_key']),
        'output_bag_size_distribution': summarize_bag_sizes(output_items, config['patient_id_depth'], config['target_key']),
        'structural_drop_count': len(structural_decisions),
        'hard_filter_drop_count': len(hard_filter_drops),
        'patient_stage_drop_count': len(patient_stage_drops),
        'deprioritized_drop_count': sum(1 for item in image_decisions_sorted if item['final_drop_reason'] == 'deprioritized_topk'),
        'kept_item_count': len(output_items),
        'dropped_item_count': sum(1 for item in image_decisions_sorted if item['final_decision'] == 'drop'),
        'drop_reason_counts': dict(Counter(
            (item['final_drop_reason'] or item['hard_drop_reason'])
            for item in image_decisions_sorted
            if item['final_decision'] == 'drop'
        )),
        'high_heterogeneity_patient_count': high_heterogeneity_patient_count,
        'high_heterogeneity_threshold': high_threshold,
        'analyzable_patient_count': analyzable_patient_count,
        'below_min_images_patient_count': below_min_images_patient_count,
        'review_patient_count': len(review_records),
        'outlier_image_count': len(outlier_records_sorted),
    }

    image_decisions_report = {
        'target_key': config['target_key'],
        'task_preset': config['name'],
        'images': image_decisions_sorted,
    }
    patients_report = {
        'target_key': config['target_key'],
        'task_preset': config['name'],
        'patients': patient_report,
    }
    review_report = {
        'target_key': config['target_key'],
        'task_preset': config['name'],
        'patients_for_review': review_records,
        'outliers': outlier_records_sorted,
    }

    return summary, image_decisions_report, patients_report, review_report


def main():
    args = parse_args()
    config = resolve_config(args)

    input_path = Path(args.input_json)
    image_root = Path(args.image_root)
    mask_root = Path(args.mask_root)
    output_path = Path(args.output_json)
    report_dir = Path(args.report_dir) if args.report_dir else output_path.parent

    if not input_path.exists():
        raise FileNotFoundError(f'Input JSON not found: {input_path}')
    if not image_root.exists():
        raise FileNotFoundError(f'Image root not found: {image_root}')
    if not mask_root.exists():
        raise FileNotFoundError(f'Mask root not found: {mask_root}')

    input_items = load_items(input_path)
    analyzed_items, structural_decisions = build_analyzed_items(
        input_items,
        image_root,
        mask_root,
        config['patient_id_depth'],
        config['target_key'],
    )
    hard_filter_candidates, hard_filter_drops = hard_filter_items(analyzed_items, config)
    patient_records, patient_review_records, patient_stage_drops = build_patient_records(
        hard_filter_candidates,
        config['target_key'],
    )
    patient_results, outlier_records = select_topk_per_patient(patient_records, config)

    image_decisions = list(structural_decisions)
    image_decisions.extend(hard_filter_drops)
    image_decisions.extend(patient_stage_drops)
    for patient_record in patient_records:
        image_decisions.extend(entry['decision'] for entry in patient_record['entries'])

    kept_entries = [
        entry
        for patient_record in patient_records
        for entry in patient_record['entries']
        if entry['decision']['final_decision'] in ('floor_keep', 'topk_keep')
    ]
    output_items = [entry['item'] for entry in sorted(kept_entries, key=lambda item: item['index'])]

    summary, image_decisions_report, patients_report, review_report = build_reports(
        input_items=input_items,
        output_items=output_items,
        structural_decisions=structural_decisions,
        hard_filter_drops=hard_filter_drops,
        patient_stage_drops=patient_stage_drops,
        image_decisions=image_decisions,
        patient_results=patient_results,
        patient_review_records=patient_review_records,
        outlier_records=outlier_records,
        config=config,
    )

    if not args.dry_run:
        save_json(output_path, output_items)
    save_json(report_dir / f'summary.{config["target_key"]}.json', summary)
    save_json(report_dir / f'image_decisions.{config["target_key"]}.json', image_decisions_report)
    save_json(report_dir / f'patients.{config["target_key"]}.json', patients_report)
    save_json(report_dir / f'review.{config["target_key"]}.json', review_report)

    print(f'Input JSON: {input_path}')
    print(f'Image root: {image_root}')
    print(f'Mask root: {mask_root}')
    print(f'Target key: {config["target_key"]}')
    print(f'Task preset: {config["name"]}')
    print(f'Dry run: {args.dry_run}')
    print('')
    print('Input image-level distribution:')
    print(f"  negative: {summary['input_image_level']['negative_count']}")
    print(f"  positive: {summary['input_image_level']['positive_count']}")
    print(f"  missing: {summary['input_image_level']['missing_count']}")
    print(f"  total: {summary['input_image_level']['item_count']}")
    print('')
    print('Output image-level distribution:')
    print(f"  negative: {summary['output_image_level']['negative_count']}")
    print(f"  positive: {summary['output_image_level']['positive_count']}")
    print(f"  missing: {summary['output_image_level']['missing_count']}")
    print(f"  total: {summary['output_image_level']['item_count']}")
    print('')
    print(f"Kept items: {summary['kept_item_count']}")
    print(f"Dropped items: {summary['dropped_item_count']}")
    print(f"High heterogeneity patients: {summary['high_heterogeneity_patient_count']}")
    print(f"Review patients: {summary['review_patient_count']}")
    print('Drop reasons:')
    for reason, count in sorted(summary['drop_reason_counts'].items()):
        print(f'  {reason}: {count}')
    print('')
    if args.dry_run:
        print('Filtered label JSON not written because --dry_run was set.')
    else:
        print(f'Filtered label JSON: {output_path}')
    print(f"Summary report: {report_dir / ('summary.' + config['target_key'] + '.json')}")
    print(f"Image decisions report: {report_dir / ('image_decisions.' + config['target_key'] + '.json')}")
    print(f"Patients report: {report_dir / ('patients.' + config['target_key'] + '.json')}")
    print(f"Review report: {report_dir / ('review.' + config['target_key'] + '.json')}")


if __name__ == '__main__':
    main()
