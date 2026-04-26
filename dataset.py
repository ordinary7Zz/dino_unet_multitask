import json
import os
import random

import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


class MultiTaskDataset(Dataset):
    def __init__(self, image_root, gt_root, label_json_path, size, mode, target_key=None):
        self.image_root = image_root
        self.gt_root = gt_root
        self.label_json_path = label_json_path
        self.target_key = target_key
        self.label_data = self._load_label_data(label_json_path)

        if self.target_key is not None:
            self.samples = self._build_samples_from_json()
            self.images = [sample['image_path'] for sample in self.samples]
            self.gts = [sample['mask_path'] for sample in self.samples]
            self.label_mapping = {}
        else:
            self.images = self._list_image_files(image_root)
            self.gts = self._list_image_files(gt_root)
            self.samples = None
            self.label_mapping = self._build_legacy_label_mapping(self.label_data)

        if len(self.images) != len(self.gts):
            raise ValueError(f"Image/mask count mismatch: {len(self.images)} images vs {len(self.gts)} masks")

        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def _load_label_data(self, label_json_path):
        if not label_json_path:
            return []
        with open(label_json_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        if isinstance(label_data, dict):
            return list(label_data.values())
        return label_data

    def _list_image_files(self, root):
        valid_suffixes = ('.jpg', '.png', '.PNG')
        return sorted(
            os.path.join(root, filename)
            for filename in os.listdir(root)
            if filename.endswith(valid_suffixes)
        )

    def _normalize_relative_path(self, path):
        return os.path.normpath(path.replace('\\', '/'))

    def _build_legacy_label_mapping(self, label_data):
        label_mapping = {}
        for item in label_data:
            filename = item.get('filename')
            if not filename:
                continue

            normalized_filename = self._normalize_relative_path(filename)
            basename = os.path.basename(normalized_filename)
            labels = {
                'malignancy': item.get('malignancy', -1),
                'tirads': item.get('tirads', -1)
            }

            label_mapping[normalized_filename] = labels
            label_mapping[basename] = labels
        return label_mapping

    def _build_samples_from_json(self):
        if not self.label_data:
            raise ValueError('label_json_path is required when target_key is provided')

        available_keys = set()
        for item in self.label_data:
            available_keys.update(item.keys())
        if self.target_key not in available_keys:
            raise KeyError(f"target_key '{self.target_key}' not found in label JSON")

        samples = []
        missing_paths = []

        for item in self.label_data:
            filename = item.get('filename')
            if not filename:
                continue

            normalized_filename = self._normalize_relative_path(filename)
            image_path = os.path.join(self.image_root, normalized_filename)
            mask_path = os.path.join(self.gt_root, normalized_filename)

            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                missing_paths.append((image_path, mask_path))
                continue

            samples.append({
                'filename': filename,
                'image_path': image_path,
                'mask_path': mask_path,
                'target': item.get(self.target_key, -1)
            })

        if missing_paths:
            preview = '\n'.join(
                f"image={image_path}, mask={mask_path}"
                for image_path, mask_path in missing_paths[:5]
            )
            raise FileNotFoundError(
                f"Missing image or mask files for {len(missing_paths)} JSON entries. First entries:\n{preview}"
            )

        if not samples:
            raise ValueError('No valid samples were built from label JSON')

        return samples

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.gts[idx]
        image = self.rgb_loader(image_path)
        label = self.binary_loader(mask_path)
        data = {'image': image, 'label': label}
        data = self.transform(data)

        if self.samples is not None:
            sample = self.samples[idx]
            data['filename'] = sample['filename']
            data['target'] = sample['target']
            return data

        filename = os.path.basename(image_path)
        class_labels = self.label_mapping.get(filename, {'malignancy': -1, 'tirads': -1})

        data['filename'] = filename
        data['malignancy'] = class_labels['malignancy']

        tirads_label = class_labels['tirads']
        if tirads_label != -1:
            tirads_label = tirads_label - 1
        data['tirads'] = tirads_label

        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
