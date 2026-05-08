import random
from collections import defaultdict

import torch

from dataset import MultiTaskDataset


def derive_patient_id(filename: str, depth: int = 2) -> str:
    normalized = filename.replace('\\', '/').strip('/')
    parts = [part for part in normalized.split('/') if part]
    if not parts:
        raise ValueError('filename is empty after normalization')
    if depth <= 0:
        raise ValueError(f'depth must be positive, got {depth}')
    return '/'.join(parts[:min(depth, len(parts))])


class PatientMultiTaskDataset(MultiTaskDataset):
    def __init__(
        self,
        image_root,
        gt_root,
        label_json_path,
        size,
        mode,
        target_key,
        patient_id_depth: int = 2,
        max_images_per_patient=None,
    ):
        if target_key is None:
            raise ValueError('target_key is required for PatientMultiTaskDataset')
        super().__init__(image_root, gt_root, label_json_path, size, mode, target_key=target_key)
        self.patient_id_depth = patient_id_depth
        self.max_images_per_patient = max_images_per_patient if mode == 'train' else None
        self.patient_groups = self._build_patient_groups(self.samples)
        self.patient_ids = sorted(self.patient_groups.keys())

    def _build_patient_groups(self, samples):
        patient_groups = defaultdict(list)
        for sample in samples:
            patient_id = derive_patient_id(sample['filename'], self.patient_id_depth)
            patient_groups[patient_id].append(sample)

        normalized_groups = {}
        for patient_id, group in patient_groups.items():
            deduped_by_filename = {}
            for sample in group:
                deduped_by_filename.setdefault(sample['filename'], sample)
            normalized_group = sorted(deduped_by_filename.values(), key=lambda item: item['filename'])
            valid_targets = {item.get('target', -1) for item in normalized_group if item.get('target', -1) in (0, 1)}
            if len(valid_targets) > 1:
                raise ValueError(f'Inconsistent targets found for patient_id={patient_id}: {sorted(valid_targets)}')
            normalized_groups[patient_id] = normalized_group

        return normalized_groups

    def __len__(self):
        return len(self.patient_ids)

    def _select_group_samples(self, group):
        if self.max_images_per_patient is None or len(group) <= self.max_images_per_patient:
            return group
        selected = random.sample(group, self.max_images_per_patient)
        return sorted(selected, key=lambda item: item['filename'])

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        group = self.patient_groups[patient_id]
        selected_group = self._select_group_samples(group)

        images = []
        labels = []
        filenames = []
        target = selected_group[0].get('target', -1)

        for sample in selected_group:
            image = self.rgb_loader(sample['image_path'])
            label = self.binary_loader(sample['mask_path'])
            data = self.transform({'image': image, 'label': label})
            images.append(data['image'])
            labels.append(data['label'])
            filenames.append(sample['filename'])

        return {
            'image': torch.stack(images, dim=0),
            'label': torch.stack(labels, dim=0),
            'target': int(target),
            'patient_id': patient_id,
            'filenames': filenames,
            'num_images': len(selected_group),
        }

    def get_patient_label_stats(self):
        neg_count = 0
        pos_count = 0
        missing_count = 0
        for patient_id in self.patient_ids:
            target = self.patient_groups[patient_id][0].get('target', -1)
            if target == 0:
                neg_count += 1
            elif target == 1:
                pos_count += 1
            else:
                missing_count += 1
        return {
            'negative': neg_count,
            'positive': pos_count,
            'missing': missing_count,
            'valid': neg_count + pos_count,
        }

    def get_bag_size_stats(self):
        bag_sizes = [len(group) for group in self.patient_groups.values()]
        if not bag_sizes:
            return {'min': 0, 'max': 0, 'mean': 0.0}
        return {
            'min': min(bag_sizes),
            'max': max(bag_sizes),
            'mean': sum(bag_sizes) / len(bag_sizes),
        }


def collate_patient_bags(batch):
    images = []
    labels = []
    targets = []
    bag_sizes = []
    patient_ids = []
    filenames = []

    for item in batch:
        images.append(item['image'])
        labels.append(item['label'])
        targets.append(item['target'])
        bag_sizes.append(item['num_images'])
        patient_ids.append(item['patient_id'])
        filenames.append(item['filenames'])

    return {
        'image': torch.cat(images, dim=0),
        'label': torch.cat(labels, dim=0),
        'target': torch.tensor(targets, dtype=torch.long),
        'bag_sizes': torch.tensor(bag_sizes, dtype=torch.long),
        'patient_ids': patient_ids,
        'filenames': filenames,
    }
