from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Select exactly one FTCPTC image per patient, using masks when available but not requiring them.'
    )
    parser.add_argument('--image_dir', type=str, required=True, help='Root directory for images')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input label JSON')
    parser.add_argument('--target_key', type=str, required=True, help='Binary target key, usually FTCPTC')
    parser.add_argument('--mask_dir', type=str, default=None, help='Optional root directory for masks')
    parser.add_argument('--output_json', type=str, required=True, help='Path to output filtered JSON')
    parser.add_argument('--patient_id_depth', type=int, default=2, help='Number of path segments used to derive patient_id')
    parser.add_argument('--report_json', type=str, default=None, help='Optional path to summary report JSON')
    return parser.parse_args()


def build_report_path(output_path: Path, report_json: str | None) -> Path:
    if report_json:
        return Path(report_json)
    return output_path.parent / f'{output_path.stem}.report.json'


def build_candidate(index: int, item: dict, image_root: Path, mask_root: Path | None, patient_id_depth: int, target_key: str):
    filename = item.get('filename')
    if not filename:
        return None

    normalized_filename = normalize_relative_path(filename)
    patient_id = derive_patient_id(filename, patient_id_depth)
    image_path = image_root / Path(normalized_filename)
    if not image_path.exists():
        return {
            'index': index,
            'item': item,
            'filename': filename,
            'normalized_filename': normalized_filename,
            'patient_id': patient_id,
            'target': item.get(target_key, -1),
            'image_exists': False,
            'image_path': str(image_path),
            'used_mask': False,
            'mask_path': None,
            'mask_error': None,
            'mask_area_ratio': None,
            'largest_component_pixels': None,
            'bbox_fill_ratio': None,
            'edge_touch_penalty': None,
            'score': None,
        }

    candidate = {
        'index': index,
        'item': item,
        'filename': filename,
        'normalized_filename': normalized_filename,
        'patient_id': patient_id,
        'target': item.get(target_key, -1),
        'image_exists': True,
        'image_path': str(image_path),
        'used_mask': False,
        'mask_path': None,
        'mask_error': None,
        'mask_area_ratio': None,
        'largest_component_pixels': None,
        'bbox_fill_ratio': None,
        'edge_touch_penalty': None,
        'score': None,
    }

    if mask_root is None:
        return candidate

    mask_path = resolve_mask_path(mask_root, normalized_filename)
    if mask_path is None or not mask_path.exists():
        return candidate

    candidate['mask_path'] = str(mask_path)

    try:
        _, mask_array = load_image_and_mask(image_path, mask_path)
        mask_stats = compute_mask_stats(mask_array)
        if mask_stats['foreground_pixels'] <= 0:
            candidate['mask_error'] = 'empty_mask'
            return candidate
        bbox_stats = compute_bbox_stats(mask_stats)
    except Exception as exc:
        candidate['mask_error'] = str(exc)
        return candidate

    candidate['used_mask'] = True
    candidate['mask_area_ratio'] = float(mask_stats['mask_area_ratio'])
    candidate['largest_component_pixels'] = int(mask_stats['largest_component_pixels'])
    candidate['bbox_fill_ratio'] = float(bbox_stats['bbox_fill_ratio'])
    candidate['edge_touch_penalty'] = float(bbox_stats['edge_touch_penalty'])
    candidate['score'] = float(score_item(mask_stats, bbox_stats))
    return candidate


def candidate_sort_key(candidate: dict):
    return (
        0 if candidate['used_mask'] else 1,
        -(candidate['score'] if candidate['score'] is not None else 0.0),
        -(candidate['largest_component_pixels'] if candidate['largest_component_pixels'] is not None else 0),
        candidate['edge_touch_penalty'] if candidate['edge_touch_penalty'] is not None else 1.0,
        candidate['normalized_filename'],
        candidate['index'],
    )


def select_one_per_patient(items, image_root: Path, mask_root: Path | None, target_key: str, patient_id_depth: int):
    all_patients = set()
    valid_label_groups = defaultdict(list)
    existing_candidate_groups = defaultdict(list)
    missing_image_records = []
    invalid_item_count = 0
    missing_filename_count = 0

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            invalid_item_count += 1
            continue

        filename = item.get('filename')
        if not filename:
            missing_filename_count += 1
            continue

        patient_id = derive_patient_id(filename, patient_id_depth)
        all_patients.add(patient_id)

        target = item.get(target_key, -1)
        if target not in (0, 1):
            continue

        valid_label_groups[patient_id].append({'index': index, 'item': item})
        candidate = build_candidate(index, item, image_root, mask_root, patient_id_depth, target_key)
        if candidate is None:
            continue
        if not candidate['image_exists']:
            missing_image_records.append({
                'patient_id': patient_id,
                'filename': filename,
                'image_path': candidate['image_path'],
            })
            continue
        existing_candidate_groups[patient_id].append(candidate)

    selected_candidates = []
    selected_patient_records = []
    skipped_patients = []

    patients_skipped_no_valid_label = 0
    patients_skipped_no_existing_image = 0
    patients_skipped_inconsistent_label = 0
    patients_with_single_candidate = 0
    patients_with_multiple_candidates = 0
    patients_with_any_usable_mask = 0
    patients_without_usable_mask = 0
    selected_with_mask_count = 0
    selected_without_mask_count = 0

    for patient_id in sorted(all_patients):
        valid_group = valid_label_groups.get(patient_id, [])
        if not valid_group:
            patients_skipped_no_valid_label += 1
            skipped_patients.append({'patient_id': patient_id, 'reason': 'no_valid_label'})
            continue

        label_values = sorted({entry['item'][target_key] for entry in valid_group})
        if len(label_values) > 1:
            patients_skipped_inconsistent_label += 1
            skipped_patients.append({
                'patient_id': patient_id,
                'reason': 'inconsistent_labels',
                'label_values': label_values,
                'candidate_count': len(valid_group),
            })
            continue

        existing_group = existing_candidate_groups.get(patient_id, [])
        if not existing_group:
            patients_skipped_no_existing_image += 1
            skipped_patients.append({
                'patient_id': patient_id,
                'reason': 'no_existing_image',
                'label': label_values[0],
                'candidate_count': len(valid_group),
            })
            continue

        if len(existing_group) == 1:
            patients_with_single_candidate += 1
        else:
            patients_with_multiple_candidates += 1

        usable_mask_count = sum(1 for candidate in existing_group if candidate['used_mask'])
        if usable_mask_count > 0:
            patients_with_any_usable_mask += 1
        else:
            patients_without_usable_mask += 1

        ranked = sorted(existing_group, key=candidate_sort_key)
        selected = ranked[0]
        selected_candidates.append(selected)

        if selected['used_mask']:
            selection_reason = 'best_mask_score'
            selected_with_mask_count += 1
        else:
            selection_reason = 'first_sorted_existing_image_no_usable_mask'
            selected_without_mask_count += 1

        selected_patient_records.append({
            'patient_id': patient_id,
            'selected_filename': selected['filename'],
            'label': selected['target'],
            'candidate_count': len(existing_group),
            'selection_reason': selection_reason,
            'used_mask': bool(selected['used_mask']),
            'score': selected['score'],
            'mask_area_ratio': selected['mask_area_ratio'],
            'largest_component_pixels': selected['largest_component_pixels'],
            'bbox_fill_ratio': selected['bbox_fill_ratio'],
            'edge_touch_penalty': selected['edge_touch_penalty'],
        })

    output_items = [candidate['item'] for candidate in sorted(selected_candidates, key=lambda item: item['index'])]
    report = {
        'target_key': target_key,
        'input_item_count': len(items),
        'valid_item_count': sum(1 for item in items if isinstance(item, dict) and item.get(target_key, -1) in (0, 1)),
        'input_patient_count': len(all_patients),
        'output_item_count': len(output_items),
        'output_patient_count': len(selected_patient_records),
        'patients_with_single_candidate': patients_with_single_candidate,
        'patients_with_multiple_candidates': patients_with_multiple_candidates,
        'patients_skipped_no_valid_label': patients_skipped_no_valid_label,
        'patients_skipped_no_existing_image': patients_skipped_no_existing_image,
        'patients_skipped_inconsistent_label': patients_skipped_inconsistent_label,
        'invalid_item_count': invalid_item_count,
        'missing_filename_count': missing_filename_count,
        'missing_image_count': len(missing_image_records),
        'patients_with_any_usable_mask': patients_with_any_usable_mask,
        'patients_without_usable_mask': patients_without_usable_mask,
        'selected_with_mask_count': selected_with_mask_count,
        'selected_without_mask_count': selected_without_mask_count,
        'class_counts_before': summarize_image_level(
            [item for item in items if isinstance(item, dict) and item.get('filename')],
            target_key,
        ),
        'class_counts_after': summarize_image_level(output_items, target_key),
        'patient_counts_before': summarize_patient_level(
            [item for item in items if isinstance(item, dict) and item.get('filename')],
            patient_id_depth,
            target_key,
        ),
        'patient_counts_after': summarize_patient_level(output_items, patient_id_depth, target_key),
        'selected_patients': selected_patient_records,
        'skipped_patients': skipped_patients,
        'missing_image_records': missing_image_records,
    }
    return output_items, report


def main():
    args = parse_args()

    image_root = Path(args.image_dir)
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    report_path = build_report_path(output_path, args.report_json)
    mask_root = Path(args.mask_dir) if args.mask_dir else None

    if not input_path.exists():
        raise FileNotFoundError(f'Input JSON not found: {input_path}')
    if not image_root.exists():
        raise FileNotFoundError(f'Image root not found: {image_root}')
    if mask_root is not None and not mask_root.exists():
        raise FileNotFoundError(f'Mask root not found: {mask_root}')
    if args.patient_id_depth <= 0:
        raise ValueError(f'patient_id_depth must be positive, got {args.patient_id_depth}')

    items = load_items(input_path)
    output_items, report = select_one_per_patient(
        items=items,
        image_root=image_root,
        mask_root=mask_root,
        target_key=args.target_key,
        patient_id_depth=args.patient_id_depth,
    )

    save_json(output_path, output_items)
    save_json(report_path, report)

    print(f'Input JSON: {input_path}')
    print(f'Image root: {image_root}')
    print(f'Mask root: {mask_root if mask_root is not None else "<not provided>"}')
    print(f'Target key: {args.target_key}')
    print(f'Output JSON: {output_path}')
    print(f'Report JSON: {report_path}')
    print(f"Output patients: {report['output_patient_count']}")
    print(f"Selected with mask: {report['selected_with_mask_count']}")
    print(f"Selected without mask: {report['selected_without_mask_count']}")


if __name__ == '__main__':
    main()
