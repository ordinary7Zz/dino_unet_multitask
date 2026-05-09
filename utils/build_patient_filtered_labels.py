from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


VALID_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_items(input_path: Path):
    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError(f'Unsupported JSON top-level type: {type(data).__name__}')


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_relative_path(path: str) -> str:
    return str(Path(path.replace('\\', '/'))).replace('\\', '/')


def derive_patient_id(filename: str, depth: int = 2) -> str:
    normalized = filename.replace('\\', '/').strip('/')
    parts = [part for part in normalized.split('/') if part]
    if not parts:
        raise ValueError('filename is empty after normalization')
    if depth <= 0:
        raise ValueError(f'depth must be positive, got {depth}')
    return '/'.join(parts[:min(depth, len(parts))])


def resolve_mask_path(mask_root: Path, normalized_filename: str) -> Path | None:
    relative_path = Path(normalized_filename)
    relative_dir = relative_path.parent
    stem = relative_path.stem
    mask_dir = mask_root / relative_dir

    if not mask_dir.is_dir():
        return None

    for candidate in mask_dir.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in VALID_IMAGE_SUFFIXES and candidate.stem == stem:
            return candidate
    return None


def load_image_and_mask(image_path: Path, mask_path: Path):
    with image_path.open('rb') as f:
        image = Image.open(f).convert('RGB')
        image_array = np.array(image)

    with mask_path.open('rb') as f:
        mask = Image.open(f).convert('L')
        mask_array = np.array(mask)

    return image_array, mask_array


def compute_mask_stats(mask_array: np.ndarray):
    foreground = mask_array > 0
    height, width = foreground.shape
    image_area = int(height * width)
    foreground_pixels = int(foreground.sum())
    mask_area_ratio = float(foreground_pixels) / float(image_area) if image_area > 0 else 0.0

    largest_component_pixels = 0
    bbox = None
    edge_touch_count = 0

    if foreground_pixels > 0:
        ys, xs = np.where(foreground)
        y_min = int(ys.min())
        y_max = int(ys.max())
        x_min = int(xs.min())
        x_max = int(xs.max())
        bbox = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'width': int(x_max - x_min + 1),
            'height': int(y_max - y_min + 1),
        }
        edge_touch_count = int(x_min == 0) + int(y_min == 0) + int(x_max == width - 1) + int(y_max == height - 1)
        largest_component_pixels = compute_largest_component_pixels(foreground)

    return {
        'image_height': int(height),
        'image_width': int(width),
        'image_area': image_area,
        'foreground_pixels': foreground_pixels,
        'mask_area_ratio': mask_area_ratio,
        'largest_component_pixels': int(largest_component_pixels),
        'bbox': bbox,
        'edge_touch_count': edge_touch_count,
    }


def compute_largest_component_pixels(foreground: np.ndarray) -> int:
    height, width = foreground.shape
    visited = np.zeros_like(foreground, dtype=bool)
    largest = 0

    for y in range(height):
        for x in range(width):
            if not foreground[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            component_size = 0

            while stack:
                cy, cx = stack.pop()
                component_size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and foreground[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if component_size > largest:
                largest = component_size

    return int(largest)


def compute_bbox_stats(mask_stats: dict):
    bbox = mask_stats.get('bbox')
    if bbox is None:
        return {
            'bbox_area_ratio': 0.0,
            'bbox_fill_ratio': 0.0,
            'edge_touch_penalty': 1.0,
        }

    bbox_area = int(bbox['width'] * bbox['height'])
    bbox_area_ratio = float(bbox_area) / float(mask_stats['image_area']) if mask_stats['image_area'] > 0 else 0.0
    bbox_fill_ratio = float(mask_stats['foreground_pixels']) / float(bbox_area) if bbox_area > 0 else 0.0
    edge_touch_penalty = float(mask_stats['edge_touch_count']) / 4.0

    return {
        'bbox_area_ratio': bbox_area_ratio,
        'bbox_fill_ratio': bbox_fill_ratio,
        'edge_touch_penalty': edge_touch_penalty,
    }


def score_item(mask_stats: dict, bbox_stats: dict) -> float:
    return (
        mask_stats['mask_area_ratio'] * 10.0
        + bbox_stats['bbox_fill_ratio'] * 1.0
        + bbox_stats['bbox_area_ratio'] * 2.0
        - bbox_stats['edge_touch_penalty'] * 0.5
    )


def summarize_image_level(items, target_key: str):
    neg = sum(1 for item in items if item.get(target_key) == 0)
    pos = sum(1 for item in items if item.get(target_key) == 1)
    missing = sum(1 for item in items if item.get(target_key, -1) not in (0, 1))
    return {
        'negative_count': neg,
        'positive_count': pos,
        'missing_count': missing,
        'valid_count': neg + pos,
        'item_count': len(items),
    }


def summarize_patient_level(items, patient_id_depth: int, target_key: str):
    patient_targets = defaultdict(set)
    patient_ids_with_items = set()

    for item in items:
        filename = item.get('filename')
        if not filename:
            continue
        patient_id = derive_patient_id(filename, patient_id_depth)
        patient_ids_with_items.add(patient_id)
        target = item.get(target_key, -1)
        if target in (0, 1):
            patient_targets[patient_id].add(int(target))
        else:
            patient_targets.setdefault(patient_id, set())

    neg = 0
    pos = 0
    missing = 0
    inconsistent = []

    for patient_id in sorted(patient_ids_with_items):
        labels = patient_targets.get(patient_id, set())
        if len(labels) == 0:
            missing += 1
        elif len(labels) == 1:
            label = next(iter(labels))
            if label == 0:
                neg += 1
            elif label == 1:
                pos += 1
        else:
            inconsistent.append({'patient_id': patient_id, 'label_values': sorted(labels)})

    return {
        'negative_count': neg,
        'positive_count': pos,
        'missing_count': missing,
        'valid_count': neg + pos,
        'patient_count': len(patient_ids_with_items),
        'inconsistent_patients': inconsistent,
    }


def analyze_items(items, image_root: Path, mask_root: Path, patient_id_depth: int):
    analyzed = []
    structural_drop_records = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            structural_drop_records.append({
                'index': index,
                'filename': None,
                'patient_id': None,
                'reason': 'invalid_item_type',
                'details': {'type': type(item).__name__},
            })
            continue

        filename = item.get('filename')
        if not filename:
            structural_drop_records.append({
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
            structural_drop_records.append({
                'index': index,
                'filename': filename,
                'patient_id': patient_id,
                'reason': 'missing_image_file',
                'details': {'image_path': str(image_path)},
            })
            continue

        if mask_path is None or not mask_path.exists():
            structural_drop_records.append({
                'index': index,
                'filename': filename,
                'patient_id': patient_id,
                'reason': 'missing_mask_file',
                'details': {'mask_root': str(mask_root)},
            })
            continue

        try:
            _, mask_array = load_image_and_mask(image_path, mask_path)
        except Exception as exc:
            structural_drop_records.append({
                'index': index,
                'filename': filename,
                'patient_id': patient_id,
                'reason': 'load_error',
                'details': {'error': str(exc)},
            })
            continue

        mask_stats = compute_mask_stats(mask_array)
        bbox_stats = compute_bbox_stats(mask_stats)
        analyzed.append({
            'index': index,
            'item': item,
            'filename': filename,
            'normalized_filename': normalized_filename,
            'patient_id': patient_id,
            'image_path': str(image_path),
            'mask_path': str(mask_path),
            'mask_stats': mask_stats,
            'bbox_stats': bbox_stats,
            'score': score_item(mask_stats, bbox_stats),
        })

    return analyzed, structural_drop_records


def filter_items(analyzed_items, structural_drop_records, target_key: str, min_mask_area_ratio: float,
                 min_largest_component_pixels: int, max_keep_per_patient: int | None,
                 review_drop_ratio: float):
    drop_records = list(structural_drop_records)
    keep_items = []
    review_patients = []
    patient_groups = defaultdict(list)

    for record in analyzed_items:
        patient_groups[record['patient_id']].append(record)

    for patient_id, group in patient_groups.items():
        valid_targets = {entry['item'].get(target_key, -1) for entry in group if entry['item'].get(target_key, -1) in (0, 1)}
        if len(valid_targets) > 1:
            review_patients.append({
                'patient_id': patient_id,
                'reasons': ['inconsistent_patient_labels'],
                'total_items': len(group),
                'kept_items': 0,
                'dropped_items': len(group),
                'label_values': sorted(valid_targets),
            })
            for entry in group:
                drop_records.append({
                    'filename': entry['filename'],
                    'patient_id': patient_id,
                    'reason': 'inconsistent_patient_labels',
                    'details': {'label_values': sorted(valid_targets)},
                })
            continue

        survivors = []
        hard_drop_count = 0
        for entry in group:
            mask_stats = entry['mask_stats']
            if mask_stats['foreground_pixels'] == 0:
                hard_drop_count += 1
                drop_records.append({
                    'filename': entry['filename'],
                    'patient_id': patient_id,
                    'reason': 'empty_mask',
                    'details': mask_stats,
                })
                continue
            if mask_stats['mask_area_ratio'] < min_mask_area_ratio:
                hard_drop_count += 1
                drop_records.append({
                    'filename': entry['filename'],
                    'patient_id': patient_id,
                    'reason': 'tiny_mask_area_ratio',
                    'details': {
                        'mask_area_ratio': mask_stats['mask_area_ratio'],
                        'threshold': min_mask_area_ratio,
                        'foreground_pixels': mask_stats['foreground_pixels'],
                        'image_area': mask_stats['image_area'],
                    },
                })
                continue
            if mask_stats['largest_component_pixels'] < min_largest_component_pixels:
                hard_drop_count += 1
                drop_records.append({
                    'filename': entry['filename'],
                    'patient_id': patient_id,
                    'reason': 'tiny_largest_component',
                    'details': {
                        'largest_component_pixels': mask_stats['largest_component_pixels'],
                        'threshold': min_largest_component_pixels,
                    },
                })
                continue
            survivors.append(entry)

        survivors.sort(key=lambda entry: (-entry['score'], entry['filename']))

        if max_keep_per_patient is not None and max_keep_per_patient > 0 and len(survivors) > max_keep_per_patient:
            kept_group = survivors[:max_keep_per_patient]
            deprioritized_group = survivors[max_keep_per_patient:]
            for rank, entry in enumerate(deprioritized_group, start=max_keep_per_patient + 1):
                drop_records.append({
                    'filename': entry['filename'],
                    'patient_id': patient_id,
                    'reason': 'deprioritized_topk',
                    'details': {
                        'score': entry['score'],
                        'rank_within_patient': rank,
                        'max_keep_per_patient': max_keep_per_patient,
                    },
                })
        else:
            kept_group = survivors
            deprioritized_group = []

        keep_items.extend(entry['item'] for entry in kept_group)

        total_items = len(group)
        dropped_items = hard_drop_count + len(deprioritized_group)
        kept_items_count = len(kept_group)
        review_reasons = []
        if kept_items_count == 0:
            review_reasons.append('all_images_dropped')
        if total_items > 0 and (float(dropped_items) / float(total_items)) >= review_drop_ratio:
            review_reasons.append('high_drop_ratio')

        if review_reasons:
            review_patients.append({
                'patient_id': patient_id,
                'reasons': review_reasons,
                'total_items': total_items,
                'kept_items': kept_items_count,
                'dropped_items': dropped_items,
            })

    return keep_items, drop_records, review_patients


def build_reports(input_items, keep_items, drop_records, review_patients, target_key: str, patient_id_depth: int,
                  min_mask_area_ratio: float, min_largest_component_pixels: int, max_keep_per_patient: int | None,
                  review_drop_ratio: float):
    summary = {
        'target_key': target_key,
        'rules': {
            'min_mask_area_ratio': min_mask_area_ratio,
            'min_largest_component_pixels': min_largest_component_pixels,
            'max_keep_per_patient': max_keep_per_patient,
            'review_drop_ratio': review_drop_ratio,
        },
        'input_image_level': summarize_image_level(input_items, target_key),
        'output_image_level': summarize_image_level(keep_items, target_key),
        'input_patient_level': summarize_patient_level(input_items, patient_id_depth, target_key),
        'output_patient_level': summarize_patient_level(keep_items, patient_id_depth, target_key),
        'drop_reason_counts': dict(Counter(record['reason'] for record in drop_records)),
        'review_patient_count': len(review_patients),
        'dropped_item_count': len(drop_records),
        'kept_item_count': len(keep_items),
    }

    drops_report = {
        'target_key': target_key,
        'rules': summary['rules'],
        'dropped_items': drop_records,
    }

    review_report = {
        'target_key': target_key,
        'patients_for_review': review_patients,
    }

    return summary, drops_report, review_report


def main():
    parser = argparse.ArgumentParser(
        description='Build a conservative filtered label JSON for patient-level training and emit keep/drop/review reports.'
    )
    parser.add_argument('--input_json', type=str, required=True, help='Path to the original label JSON file')
    parser.add_argument('--image_root', type=str, required=True, help='Path to the image root directory')
    parser.add_argument('--mask_root', type=str, required=True, help='Path to the mask root directory')
    parser.add_argument('--output_json', type=str, required=True, help='Path to save the filtered label JSON file')
    parser.add_argument('--target_key', type=str, required=True, help='Binary target key to analyze, e.g. FTCPTC')
    parser.add_argument('--patient_id_depth', type=int, default=2, help='Number of path segments used to derive patient_id')
    parser.add_argument('--min_mask_area_ratio', type=float, default=0.0005, help='Minimum foreground area ratio to keep a sample')
    parser.add_argument('--min_largest_component_pixels', type=int, default=64,
                        help='Minimum largest connected-component size in pixels to keep a sample')
    parser.add_argument('--max_keep_per_patient', type=int, default=None,
                        help='If provided, keep only top-k scored images per patient after hard filtering')
    parser.add_argument('--review_drop_ratio', type=float, default=0.8,
                        help='Flag patient for review if dropped_items / total_items >= this ratio')
    parser.add_argument('--report_dir', type=str, default=None, help='Directory to save summary/drop/review reports')
    parser.add_argument('--dry_run', action='store_true', help='Analyze and write reports without saving filtered label JSON')
    args = parser.parse_args()

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
    analyzed_items, structural_drop_records = analyze_items(input_items, image_root, mask_root, args.patient_id_depth)
    keep_items, drop_records, review_patients = filter_items(
        analyzed_items,
        structural_drop_records,
        args.target_key,
        args.min_mask_area_ratio,
        args.min_largest_component_pixels,
        args.max_keep_per_patient,
        args.review_drop_ratio,
    )
    summary, drops_report, review_report = build_reports(
        input_items,
        keep_items,
        drop_records,
        review_patients,
        args.target_key,
        args.patient_id_depth,
        args.min_mask_area_ratio,
        args.min_largest_component_pixels,
        args.max_keep_per_patient,
        args.review_drop_ratio,
    )

    if not args.dry_run:
        save_json(output_path, keep_items)
    save_json(report_dir / f'summary.{args.target_key}.json', summary)
    save_json(report_dir / f'drops.{args.target_key}.json', drops_report)
    save_json(report_dir / f'review.{args.target_key}.json', review_report)

    print(f'Input JSON: {input_path}')
    print(f'Image root: {image_root}')
    print(f'Mask root: {mask_root}')
    print(f'Target key: {args.target_key}')
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
    print('Patient-level distribution before filtering:')
    print(f"  negative: {summary['input_patient_level']['negative_count']}")
    print(f"  positive: {summary['input_patient_level']['positive_count']}")
    print(f"  missing: {summary['input_patient_level']['missing_count']}")
    print(f"  total: {summary['input_patient_level']['patient_count']}")
    print('')
    print('Patient-level distribution after filtering:')
    print(f"  negative: {summary['output_patient_level']['negative_count']}")
    print(f"  positive: {summary['output_patient_level']['positive_count']}")
    print(f"  missing: {summary['output_patient_level']['missing_count']}")
    print(f"  total: {summary['output_patient_level']['patient_count']}")
    print('')
    print(f"Dropped items: {summary['dropped_item_count']}")
    print(f"Review patients: {summary['review_patient_count']}")
    print('Drop reasons:')
    for reason, count in sorted(summary['drop_reason_counts'].items()):
        print(f'  {reason}: {count}')
    print('')
    if args.dry_run:
        print('Filtered label JSON not written because --dry_run was set.')
    else:
        print(f'Filtered label JSON: {output_path}')
    print(f"Summary report: {report_dir / f'summary.{args.target_key}.json'}")
    print(f"Drop report: {report_dir / f'drops.{args.target_key}.json'}")
    print(f"Review report: {report_dir / f'review.{args.target_key}.json'}")


if __name__ == '__main__':
    main()
