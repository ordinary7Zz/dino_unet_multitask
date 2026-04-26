import torchvision.transforms.functional as F
import numpy as np
import random
import os
import json
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


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
    def __init__(self, image_root, gt_root, label_json_path, size, mode):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.PNG')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.PNG')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        
        # Load classification labels from JSON file
        with open(label_json_path, 'r') as f:
            self.label_data = json.load(f)
        
        # Create a mapping from image filename to classification labels
        self.label_mapping = {}
        for item in self.label_data:
            filename = item.get('filename')
            if filename:
                self.label_mapping[filename] = {
                    'malignancy': item.get('malignancy', -1),
                    'tirads': item.get('tirads', -1)
                }
        
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

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        
        # Get classification labels
        filename = os.path.basename(self.images[idx])
        class_labels = self.label_mapping.get(filename, {'malignancy': -1, 'tirads': -1})
        
        # Add classification labels to the data
        data['filename'] = filename
        data['malignancy'] = class_labels['malignancy']
        
        # Convert TIRADS label from [1-5] to [0-4] to match PyTorch CrossEntropyLoss requirements
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
