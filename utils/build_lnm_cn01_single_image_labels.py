from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

try:
    from build_patient_filtered_labels import (
        derive_patient_id,
        load_items,
        normalize_relative_path,
        save_json,
        summarize_image_level,
        summarize_patient_level,
    )
except ImportError:
    from utils.build_patient_filtered_labels import (
        derive_patient_id,
        load_items,
        normalize_relative_path,
        save_json,
        summarize_image_level,
        summarize_patient_level,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Select exactly one LNM_CN01 image per patient without depending on masks.'
    )
    parser.add_argument('--image_dir', type=str, required=True, help='Root directory for images')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input label JSON')
    parser.add_argument('--target_key', type=str, required=True, help='Binary target key, usually LNM_CN01')
    parser.add_argument('--mask_dir', type=str, default=None, help='Optional mask root, accepted but ignored for selection')
    parser.add_argument('--output_json', type=str, required=True, help='Path to output filtered JSON')
    parser.add_argument('--patient_id_depth', type=int, default=2, help='Number of path segments used to derive patient_id')
    parser.add_argument('--report_json', type=str, default=None, help='Optional path to summary report JSON')
    return parser.parse_args()


def build_report_path(output_path: Path, report_json: str | None) -> Path:
    if report_json:
        return Path(report_json)
    return output_path.parent / f'{output_path.stem}.report.json'


def select_one_per_patient(items, image_root: Path, target_key: str, patient_id_depth: int, mask_dir: str | None):
    all_patients = set()
    valid_label_groups = defaultdict(list)
    missing_image_records = []
    skipped_patients = []
    selected_patients = []
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

        normalized_filename = normalize_relative_path(filename)
        image_path = image_root / Path(normalized_filename)
        valid_label_groups[patient_id].append({
            'index': index,
            'item': item,
            'filename': filename,
            'normalized_filename': normalized_filename,
            'patient_id': patient_id,
            'target': target,
            'image_exists': image_path.exists(),
            'image_path': str(image_path),
        })
        if not image_path.exists():
            missing_image_records.append({
                'patient_id': patient_id,
                'filename': filename,
                'image_path': str(image_path),
            })

    selected_candidates = []
    patients_skipped_no_valid_label = 0
    patients_skipped_no_existing_image = 0
    patients_skipped_inconsistent_label = 0
    patients_with_single_candidate = 0
    patients_with_multiple_candidates = 0

    for patient_id in sorted(all_patients):
        group = valid_label_groups.get(patient_id, [])
        if not group:
            patients_skipped_no_valid_label += 1
            skipped_patients.append({'patient_id': patient_id, 'reason': 'no_valid_label'})
            continue

        label_values = sorted({entry['target'] for entry in group})
        if len(label_values) > 1:
            patients_skipped_inconsistent_label += 1
            skipped_patients.append({
                'patient_id': patient_id,
                'reason': 'inconsistent_labels',
                'label_values': label_values,
                'candidate_count': len(group),
            })
            continue

        existing_group = [entry for entry in group if entry['image_exists']]
        if not existing_group:
            patients_skipped_no_existing_image += 1
            skipped_patients.append({
                'patient_id': patient_id,
                'reason': 'no_existing_image',
                'label': label_values[0],
                'candidate_count': len(group),
            })
            continue

        if len(existing_group) == 1:
            patients_with_single_candidate += 1
        else:
            patients_with_multiple_candidates += 1

        ranked = sorted(existing_group, key=lambda entry: (entry['normalized_filename'], entry['index']))
        selected = ranked[0]
        selected_candidates.append(selected)
        selected_patients.append({
            'patient_id': patient_id,
            'selected_filename': selected['filename'],
            'label': selected['target'],
            'candidate_count': len(existing_group),
            'selection_reason': 'first_sorted_existing_image',
            'mask_ignored_for_selection': mask_dir is not None,
        })

    output_items = [candidate['item'] for candidate in sorted(selected_candidates, key=lambda item: item['index'])]
    report = {
        'target_key': target_key,
        'input_item_count': len(items),
        'valid_item_count': sum(1 for item in items if isinstance(item, dict) and item.get(target_key, -1) in (0, 1)),
        'input_patient_count': len(all_patients),
        'output_item_count': len(output_items),
        'output_patient_count': len(selected_patients),
        'patients_with_single_candidate': patients_with_single_candidate,
        'patients_with_multiple_candidates': patients_with_multiple_candidates,
        'patients_skipped_no_valid_label': patients_skipped_no_valid_label,
        'patients_skipped_no_existing_image': patients_skipped_no_existing_image,
        'patients_skipped_inconsistent_label': patients_skipped_inconsistent_label,
        'invalid_item_count': invalid_item_count,
        'missing_filename_count': missing_filename_count,
        'missing_image_count': len(missing_image_records),
        'mask_ignored_for_selection': mask_dir is not None,
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
        'selected_patients': selected_patients,
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

    if not input_path.exists():
        raise FileNotFoundError(f'Input JSON not found: {input_path}')
    if not image_root.exists():
        raise FileNotFoundError(f'Image root not found: {image_root}')
    if args.patient_id_depth <= 0:
        raise ValueError(f'patient_id_depth must be positive, got {args.patient_id_depth}')

    items = load_items(input_path)
    output_items, report = select_one_per_patient(
        items=items,
        image_root=image_root,
        target_key=args.target_key,
        patient_id_depth=args.patient_id_depth,
        mask_dir=args.mask_dir,
    )

    save_json(output_path, output_items)
    save_json(report_path, report)

    print(f'Input JSON: {input_path}')
    print(f'Image root: {image_root}')
    print(f'Mask root ignored for selection: {args.mask_dir if args.mask_dir is not None else "<not provided>"}')
    print(f'Target key: {args.target_key}')
    print(f'Output JSON: {output_path}')
    print(f'Report JSON: {report_path}')
    print(f"Output patients: {report['output_patient_count']}")


if __name__ == '__main__':
    main()
