from __future__ import annotations

import argparse
import json
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
        summarize_patient_level,
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
        summarize_patient_level,
    )


DEFAULT_HETEROGENEITY_FEATURES = [
    'mask_area_ratio',
    'bbox_fill_ratio',
    'bbox_area_ratio',
    'largest_component_ratio',
    'center_x_ratio',
    'center_y_ratio',
    'edge_touch_penalty',
    'masked_intensity_mean',
    'masked_intensity_std',
]

DEFAULT_OUTLIER_FEATURES = [
    'mask_area_ratio',
    'bbox_fill_ratio',
    'largest_component_ratio',
    'edge_touch_penalty',
    'center_x_ratio',
    'center_y_ratio',
    'masked_intensity_mean',
]

UNIT_INTERVAL_FEATURES = {
    'mask_area_ratio',
    'largest_component_ratio',
    'bbox_area_ratio',
    'bbox_fill_ratio',
    'center_x_ratio',
    'center_y_ratio',
    'edge_touch_penalty',
    'component_fragmentation',
    'occupancy_irregularity',
}

INTENSITY_FEATURES = {
    'masked_intensity_mean',
    'masked_intensity_std',
    'masked_intensity_p10',
    'masked_intensity_p50',
    'masked_intensity_p90',
    'bbox_intensity_mean',
    'foreground_background_mean_gap',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze within-patient image heterogeneity and emit auditable JSON reports.'
    )
    parser.add_argument('--input_json', type=str, required=True, help='Path to the original label JSON file')
    parser.add_argument('--image_root', type=str, required=True, help='Path to the image root directory')
    parser.add_argument('--mask_root', type=str, required=True, help='Path to the mask root directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save analysis reports')
    parser.add_argument('--target_key', type=str, default=None, help='Optional binary target key for label audits')
    parser.add_argument('--patient_id_depth', type=int, default=2, help='Number of path segments used to derive patient_id')
    parser.add_argument('--min_images_per_patient', type=int, default=2,
                        help='Minimum valid images per patient required to compute heterogeneity metrics')
    parser.add_argument('--heterogeneity_features', type=str, default=None,
                        help='Comma-separated feature list; defaults to a conservative built-in set')
    parser.add_argument('--outlier_z_threshold', type=float, default=2.5,
                        help='Absolute z-score threshold when MAD-based scoring is unavailable')
    parser.add_argument('--outlier_mad_threshold', type=float, default=3.5,
                        help='Absolute modified-z threshold when MAD-based scoring is available')
    parser.add_argument('--high_heterogeneity_percentile', type=float, default=90.0,
                        help='Percentile cutoff used to flag high-heterogeneity patients')
    parser.add_argument('--max_patients', type=int, default=None,
                        help='Optional debug limit on the number of sorted patient groups to analyze')
    parser.add_argument('--report_prefix', type=str, default=None, help='Optional prefix prepended to report filenames')
    return parser.parse_args()


def parse_feature_list(raw_value: str | None):
    if raw_value is None:
        return list(DEFAULT_HETEROGENEITY_FEATURES)
    features = [part.strip() for part in raw_value.split(',') if part.strip()]
    if not features:
        raise ValueError('heterogeneity_features is empty after parsing')
    return features


def build_report_path(output_dir: Path, base_name: str, target_key: str | None, report_prefix: str | None):
    parts = []
    if report_prefix:
        parts.append(report_prefix)
    parts.append(base_name)
    if target_key:
        parts.append(target_key)
    return output_dir / f"{'.'.join(parts)}.json"


def to_float(value):
    if value is None:
        return None
    return float(value)


def compute_numeric_summary(values):
    if not values:
        return None

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None

    mean = float(array.mean())
    median = float(np.median(array))
    std = float(array.std())
    value_min = float(array.min())
    value_max = float(array.max())
    iqr = float(np.percentile(array, 75) - np.percentile(array, 25))
    mad = float(np.median(np.abs(array - median)))
    cv = None if abs(mean) < 1e-12 else float(std / abs(mean))

    return {
        'count': int(array.size),
        'mean': mean,
        'median': median,
        'std': std,
        'min': value_min,
        'max': value_max,
        'range': float(value_max - value_min),
        'iqr': iqr,
        'cv': cv,
        'mad': mad,
    }


def normalized_dispersion(feature_name: str, summary: dict | None):
    if not summary or summary['count'] < 2:
        return None

    if feature_name in UNIT_INTERVAL_FEATURES:
        return float(summary['iqr'])
    if feature_name in INTENSITY_FEATURES:
        return float(summary['iqr']) / 255.0

    denominator = max(abs(summary['median']), 1.0)
    return float(summary['iqr']) / denominator


def summarize_scores(values):
    if not values:
        return {
            'count': 0,
            'min': None,
            'median': None,
            'mean': None,
            'p90': None,
            'p95': None,
            'max': None,
        }

    array = np.asarray(values, dtype=np.float64)
    return {
        'count': int(array.size),
        'min': float(array.min()),
        'median': float(np.median(array)),
        'mean': float(array.mean()),
        'p90': float(np.percentile(array, 90)),
        'p95': float(np.percentile(array, 95)),
        'max': float(array.max()),
    }


def extract_image_features(image_array: np.ndarray, mask_array: np.ndarray, mask_stats: dict, bbox_stats: dict):
    foreground = mask_array > 0
    grayscale = image_array.astype(np.float32).mean(axis=2) if image_array.ndim == 3 else image_array.astype(np.float32)
    foreground_pixels = mask_stats['foreground_pixels']
    largest_component_pixels = mask_stats['largest_component_pixels']
    largest_component_ratio = (
        float(largest_component_pixels) / float(foreground_pixels)
        if foreground_pixels > 0 else 0.0
    )

    bbox = mask_stats.get('bbox')
    bbox_aspect_ratio = None
    center_x_ratio = None
    center_y_ratio = None
    bbox_intensity_mean = None
    foreground_background_mean_gap = None

    if bbox is not None:
        bbox_aspect_ratio = float(bbox['width']) / float(max(bbox['height'], 1))
        center_x = (float(bbox['x_min']) + float(bbox['x_max'])) / 2.0
        center_y = (float(bbox['y_min']) + float(bbox['y_max'])) / 2.0
        center_x_ratio = center_x / float(max(mask_stats['image_width'] - 1, 1))
        center_y_ratio = center_y / float(max(mask_stats['image_height'] - 1, 1))

        y0, y1 = bbox['y_min'], bbox['y_max'] + 1
        x0, x1 = bbox['x_min'], bbox['x_max'] + 1
        bbox_gray = grayscale[y0:y1, x0:x1]
        bbox_fg = foreground[y0:y1, x0:x1]
        if bbox_gray.size > 0:
            bbox_intensity_mean = float(bbox_gray.mean())
        if bbox_fg.size > 0 and np.any(~bbox_fg):
            background_values = bbox_gray[~bbox_fg]
            foreground_values_local = bbox_gray[bbox_fg]
            if foreground_values_local.size > 0 and background_values.size > 0:
                foreground_background_mean_gap = float(foreground_values_local.mean() - background_values.mean())

    masked_values = grayscale[foreground]
    if masked_values.size == 0:
        masked_intensity_mean = None
        masked_intensity_std = None
        masked_intensity_p10 = None
        masked_intensity_p50 = None
        masked_intensity_p90 = None
    else:
        masked_intensity_mean = float(masked_values.mean())
        masked_intensity_std = float(masked_values.std())
        masked_intensity_p10 = float(np.percentile(masked_values, 10))
        masked_intensity_p50 = float(np.percentile(masked_values, 50))
        masked_intensity_p90 = float(np.percentile(masked_values, 90))

    return {
        'foreground_pixels': float(foreground_pixels),
        'mask_area_ratio': float(mask_stats['mask_area_ratio']),
        'largest_component_pixels': float(largest_component_pixels),
        'largest_component_ratio': largest_component_ratio,
        'bbox_area_ratio': float(bbox_stats['bbox_area_ratio']),
        'bbox_fill_ratio': float(bbox_stats['bbox_fill_ratio']),
        'edge_touch_penalty': float(bbox_stats['edge_touch_penalty']),
        'bbox_aspect_ratio': to_float(bbox_aspect_ratio),
        'center_x_ratio': to_float(center_x_ratio),
        'center_y_ratio': to_float(center_y_ratio),
        'component_fragmentation': float(1.0 - largest_component_ratio),
        'occupancy_irregularity': float(1.0 - bbox_stats['bbox_fill_ratio']),
        'masked_intensity_mean': to_float(masked_intensity_mean),
        'masked_intensity_std': to_float(masked_intensity_std),
        'masked_intensity_p10': to_float(masked_intensity_p10),
        'masked_intensity_p50': to_float(masked_intensity_p50),
        'masked_intensity_p90': to_float(masked_intensity_p90),
        'bbox_intensity_mean': to_float(bbox_intensity_mean),
        'foreground_background_mean_gap': to_float(foreground_background_mean_gap),
    }


def build_analyzed_items(items, image_root: Path, mask_root: Path, patient_id_depth: int, target_key: str | None):
    analyzed_items = []
    audit_drop_records = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            audit_drop_records.append({
                'index': index,
                'filename': None,
                'patient_id': None,
                'reason': 'invalid_item_type',
                'details': {'type': type(item).__name__},
            })
            continue

        filename = item.get('filename')
        if not filename:
            audit_drop_records.append({
                'index': index,
                'filename': None,
                'patient_id': None,
                'reason': 'missing_filename',
                'details': {},
            })
            continue

        normalized_filename = normalize_relative_path(filename)
        patient_id = derive_patient_id(filename, patient_id_depth)
        image_path = image_root / Path(normalized_filename)
        mask_path = resolve_mask_path(mask_root, normalized_filename)

        if not image_path.exists():
            audit_drop_records.append({
                'index': index,
                'filename': filename,
                'patient_id': patient_id,
                'reason': 'missing_image_file',
                'details': {'image_path': str(image_path)},
            })
            continue

        if mask_path is None or not mask_path.exists():
            audit_drop_records.append({
                'index': index,
                'filename': filename,
                'patient_id': patient_id,
                'reason': 'missing_mask_file',
                'details': {'mask_root': str(mask_root)},
            })
            continue

        try:
            image_array, mask_array = load_image_and_mask(image_path, mask_path)
        except Exception as exc:
            audit_drop_records.append({
                'index': index,
                'filename': filename,
                'patient_id': patient_id,
                'reason': 'load_error',
                'details': {'error': str(exc)},
            })
            continue

        mask_stats = compute_mask_stats(mask_array)
        if mask_stats['foreground_pixels'] == 0:
            audit_drop_records.append({
                'index': index,
                'filename': filename,
                'patient_id': patient_id,
                'reason': 'empty_mask',
                'details': mask_stats,
            })
            continue

        bbox_stats = compute_bbox_stats(mask_stats)
        features = extract_image_features(image_array, mask_array, mask_stats, bbox_stats)
        analyzed_items.append({
            'index': index,
            'item': item,
            'filename': filename,
            'normalized_filename': normalized_filename,
            'patient_id': patient_id,
            'target': item.get(target_key, -1) if target_key else None,
            'image_path': str(image_path),
            'mask_path': str(mask_path),
            'mask_stats': mask_stats,
            'bbox_stats': bbox_stats,
            'features': features,
        })

    return analyzed_items, audit_drop_records


def group_patient_records(analyzed_items, patient_id_depth: int, target_key: str | None, max_patients: int | None):
    patient_groups = defaultdict(list)
    for entry in analyzed_items:
        patient_id = derive_patient_id(entry['filename'], patient_id_depth)
        patient_groups[patient_id].append(entry)

    patient_ids = sorted(patient_groups.keys())
    if max_patients is not None:
        patient_ids = patient_ids[:max_patients]

    grouped = []
    for patient_id in patient_ids:
        group = patient_groups[patient_id]
        deduped_by_filename = {}
        duplicate_filenames = []
        for entry in group:
            if entry['filename'] in deduped_by_filename:
                duplicate_filenames.append(entry['filename'])
                continue
            deduped_by_filename[entry['filename']] = entry

        entries = sorted(deduped_by_filename.values(), key=lambda item: item['filename'])
        valid_targets = {
            entry['item'].get(target_key, -1)
            for entry in entries
            if target_key and entry['item'].get(target_key, -1) in (0, 1)
        }

        audit_flags = []
        if duplicate_filenames:
            audit_flags.append('duplicate_filename')
        if len(valid_targets) > 1:
            audit_flags.append('inconsistent_patient_labels')

        grouped.append({
            'patient_id': patient_id,
            'entries': entries,
            'duplicate_filenames': sorted(set(duplicate_filenames)),
            'label_values': sorted(valid_targets),
            'audit_flags': audit_flags,
            'target': next(iter(valid_targets)) if len(valid_targets) == 1 else None,
        })

    return grouped


def compute_patient_feature_summary(entries, feature_names):
    summary = {}
    for feature_name in feature_names:
        values = [entry['features'].get(feature_name) for entry in entries if entry['features'].get(feature_name) is not None]
        feature_summary = compute_numeric_summary(values)
        if feature_summary is None:
            continue
        summary[feature_name] = feature_summary
    return summary


def compute_patient_heterogeneity(entries, feature_names, min_images_per_patient: int):
    if len(entries) < min_images_per_patient:
        return None, {}, 'below_min_images'

    feature_summary = compute_patient_feature_summary(entries, feature_names)
    per_feature_dispersion = {}
    normalized_scores = []

    for feature_name, summary in feature_summary.items():
        if summary['count'] < 2:
            continue
        dispersion = normalized_dispersion(feature_name, summary)
        per_feature_dispersion[feature_name] = {
            'count': summary['count'],
            'std': summary['std'],
            'iqr': summary['iqr'],
            'cv': summary['cv'],
            'mad': summary['mad'],
            'normalized_dispersion': dispersion,
        }
        if dispersion is not None:
            normalized_scores.append(dispersion)

    if not normalized_scores:
        return None, per_feature_dispersion, 'insufficient_feature_variation'

    heterogeneity_score = float(np.mean(normalized_scores))
    return heterogeneity_score, per_feature_dispersion, None


def compute_feature_score(value, values, z_threshold: float, mad_threshold: float):
    if value is None:
        return None

    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if array.size < 2:
        return None

    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad > 1e-12:
        score = 0.6745 * (float(value) - median) / mad
        method = 'mad'
        threshold = mad_threshold
        reference_center = median
        reference_dispersion = mad
    else:
        mean = float(array.mean())
        std = float(array.std())
        if std <= 1e-12:
            return None
        score = (float(value) - mean) / std
        method = 'zscore'
        threshold = z_threshold
        reference_center = mean
        reference_dispersion = std

    return {
        'score': float(score),
        'abs_score': float(abs(score)),
        'method': method,
        'threshold': float(threshold),
        'reference_center': float(reference_center),
        'reference_dispersion': float(reference_dispersion),
    }


def detect_within_patient_outliers(patient_record, feature_names, z_threshold: float, mad_threshold: float):
    entries = patient_record['entries']
    if len(entries) < 2:
        return []

    feature_value_map = {
        feature_name: [entry['features'].get(feature_name) for entry in entries]
        for feature_name in feature_names
    }

    outliers = []
    for entry in entries:
        triggered_features = []
        feature_values = {}
        reference_patient_medians = {}
        reference_patient_dispersion = {}
        reasons = []
        outlier_score = 0.0

        for feature_name in feature_names:
            value = entry['features'].get(feature_name)
            feature_values[feature_name] = value
            score_info = compute_feature_score(value, feature_value_map[feature_name], z_threshold, mad_threshold)
            if score_info is None:
                continue

            reference_patient_medians[feature_name] = score_info['reference_center']
            reference_patient_dispersion[feature_name] = score_info['reference_dispersion']

            if score_info['abs_score'] >= score_info['threshold']:
                triggered_features.append(feature_name)
                outlier_score = max(outlier_score, score_info['abs_score'])
                reasons.append(
                    f"{feature_name}:{score_info['method']}={score_info['score']:.3f}"
                )

        if triggered_features:
            outliers.append({
                'patient_id': patient_record['patient_id'],
                'filename': entry['filename'],
                'outlier_score': float(outlier_score),
                'triggered_features': triggered_features,
                'feature_values': feature_values,
                'reference_patient_medians': reference_patient_medians,
                'reference_patient_dispersion': reference_patient_dispersion,
                'reasons': reasons,
            })

    return outliers


def build_reports(input_items, analyzed_items, audit_drop_records, patient_records, patient_results, outlier_records,
                  target_key: str | None, patient_id_depth: int, min_images_per_patient: int,
                  heterogeneity_features, outlier_z_threshold: float, outlier_mad_threshold: float,
                  high_heterogeneity_percentile: float, max_patients: int | None):
    analyzable_scores = [
        result['heterogeneity_score']
        for result in patient_results
        if result['heterogeneity_score'] is not None
    ]
    score_distribution = summarize_scores(analyzable_scores)

    high_threshold = None
    if analyzable_scores:
        high_threshold = float(np.percentile(np.asarray(analyzable_scores, dtype=np.float64), high_heterogeneity_percentile))

    high_patient_count = 0
    patients_report = []
    patient_audit_records = []
    outlier_by_patient = defaultdict(list)
    for outlier in outlier_records:
        outlier_by_patient[outlier['patient_id']].append(outlier['filename'])

    for result in patient_results:
        audit_flags = list(result['audit_flags'])
        if result['heterogeneity_error']:
            audit_flags.append(result['heterogeneity_error'])

        high_flag = (
            high_threshold is not None
            and result['heterogeneity_score'] is not None
            and result['heterogeneity_score'] >= high_threshold
        )
        if high_flag:
            high_patient_count += 1

        patients_report.append({
            'patient_id': result['patient_id'],
            'image_count': result['image_count'],
            'valid_image_count': result['valid_image_count'],
            'filenames': result['filenames'],
            'target': result['target'],
            'heterogeneity_score': result['heterogeneity_score'],
            'high_heterogeneity_flag': high_flag,
            'per_feature_dispersion': result['per_feature_dispersion'],
            'feature_summary': result['feature_summary'],
            'outlier_count': len(outlier_by_patient[result['patient_id']]),
            'outlier_filenames': sorted(set(outlier_by_patient[result['patient_id']])),
            'audit_flags': audit_flags,
        })

        if audit_flags or result['duplicate_filenames'] or result['label_values']:
            patient_audit_records.append({
                'patient_id': result['patient_id'],
                'audit_flags': audit_flags,
                'duplicate_filenames': result['duplicate_filenames'],
                'label_values': result['label_values'],
                'valid_image_count': result['valid_image_count'],
            })

    summary = {
        'target_key': target_key,
        'config': {
            'patient_id_depth': patient_id_depth,
            'min_images_per_patient': min_images_per_patient,
            'heterogeneity_features': list(heterogeneity_features),
            'outlier_z_threshold': outlier_z_threshold,
            'outlier_mad_threshold': outlier_mad_threshold,
            'high_heterogeneity_percentile': high_heterogeneity_percentile,
            'max_patients': max_patients,
        },
        'input_item_count': len(input_items),
        'valid_analyzed_item_count': len(analyzed_items),
        'audit_drop_count': len(audit_drop_records),
        'audit_drop_reason_counts': dict(Counter(record['reason'] for record in audit_drop_records)),
        'total_patient_count': len(patient_records),
        'analyzable_patient_count': sum(1 for result in patient_results if result['heterogeneity_score'] is not None),
        'below_min_images_patient_count': sum(1 for result in patient_results if result['heterogeneity_error'] == 'below_min_images'),
        'high_heterogeneity_patient_count': high_patient_count,
        'high_heterogeneity_threshold': high_threshold,
        'heterogeneity_score_distribution': score_distribution,
    }
    if target_key:
        summary['input_patient_level'] = summarize_patient_level(input_items, patient_id_depth, target_key)

    patients_report.sort(key=lambda item: (
        item['heterogeneity_score'] is None,
        -(item['heterogeneity_score'] or -1.0),
        item['patient_id'],
    ))
    outlier_records.sort(key=lambda item: (-item['outlier_score'], item['patient_id'], item['filename']))
    patient_audit_records.sort(key=lambda item: item['patient_id'])

    return (
        summary,
        {'target_key': target_key, 'patients': patients_report},
        {'target_key': target_key, 'outliers': outlier_records},
        {
            'target_key': target_key,
            'dropped_items': audit_drop_records,
            'patients': patient_audit_records,
        },
    )


def main():
    args = parse_args()

    input_path = Path(args.input_json)
    image_root = Path(args.image_root)
    mask_root = Path(args.mask_root)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f'Input JSON not found: {input_path}')
    if not image_root.exists():
        raise FileNotFoundError(f'Image root not found: {image_root}')
    if not mask_root.exists():
        raise FileNotFoundError(f'Mask root not found: {mask_root}')
    if args.patient_id_depth <= 0:
        raise ValueError(f'patient_id_depth must be positive, got {args.patient_id_depth}')
    if args.min_images_per_patient <= 0:
        raise ValueError(f'min_images_per_patient must be positive, got {args.min_images_per_patient}')

    heterogeneity_features = parse_feature_list(args.heterogeneity_features)
    outlier_features = [feature for feature in DEFAULT_OUTLIER_FEATURES if feature in heterogeneity_features]
    if not outlier_features:
        outlier_features = list(heterogeneity_features)

    input_items = load_items(input_path)
    analyzed_items, audit_drop_records = build_analyzed_items(
        input_items,
        image_root,
        mask_root,
        args.patient_id_depth,
        args.target_key,
    )
    patient_records = group_patient_records(
        analyzed_items,
        args.patient_id_depth,
        args.target_key,
        args.max_patients,
    )

    patient_results = []
    outlier_records = []
    for patient_record in patient_records:
        feature_summary = compute_patient_feature_summary(patient_record['entries'], heterogeneity_features)
        heterogeneity_score, per_feature_dispersion, heterogeneity_error = compute_patient_heterogeneity(
            patient_record['entries'],
            heterogeneity_features,
            args.min_images_per_patient,
        )
        patient_result = {
            'patient_id': patient_record['patient_id'],
            'image_count': len(patient_record['entries']),
            'valid_image_count': len(patient_record['entries']),
            'filenames': [entry['filename'] for entry in patient_record['entries']],
            'target': patient_record['target'],
            'duplicate_filenames': patient_record['duplicate_filenames'],
            'label_values': patient_record['label_values'],
            'audit_flags': patient_record['audit_flags'],
            'feature_summary': feature_summary,
            'per_feature_dispersion': per_feature_dispersion,
            'heterogeneity_score': heterogeneity_score,
            'heterogeneity_error': heterogeneity_error,
        }
        patient_results.append(patient_result)
        outlier_records.extend(
            detect_within_patient_outliers(
                patient_record,
                outlier_features,
                args.outlier_z_threshold,
                args.outlier_mad_threshold,
            )
        )

    summary, patients_report, outliers_report, audit_report = build_reports(
        input_items,
        analyzed_items,
        audit_drop_records,
        patient_records,
        patient_results,
        outlier_records,
        args.target_key,
        args.patient_id_depth,
        args.min_images_per_patient,
        heterogeneity_features,
        args.outlier_z_threshold,
        args.outlier_mad_threshold,
        args.high_heterogeneity_percentile,
        args.max_patients,
    )

    save_json(build_report_path(output_dir, 'summary', args.target_key, args.report_prefix), summary)
    save_json(build_report_path(output_dir, 'patients', args.target_key, args.report_prefix), patients_report)
    save_json(build_report_path(output_dir, 'outliers', args.target_key, args.report_prefix), outliers_report)
    save_json(build_report_path(output_dir, 'audit', args.target_key, args.report_prefix), audit_report)

    print(f'Input JSON: {input_path}')
    print(f'Image root: {image_root}')
    print(f'Mask root: {mask_root}')
    print(f'Output dir: {output_dir}')
    print(f'Target key: {args.target_key}')
    print(f'Valid analyzed items: {summary["valid_analyzed_item_count"]}')
    print(f'Audit drops: {summary["audit_drop_count"]}')
    print(f'Total patients: {summary["total_patient_count"]}')
    print(f'Analyzable patients: {summary["analyzable_patient_count"]}')
    print(f'High heterogeneity patients: {summary["high_heterogeneity_patient_count"]}')
    print('Audit drop reasons:')
    for reason, count in sorted(summary['audit_drop_reason_counts'].items()):
        print(f'  {reason}: {count}')


if __name__ == '__main__':
    main()
