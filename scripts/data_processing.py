import json
import os
import torch
import torch.utils.data
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import numpy as np
import random
import shutil

# Custom dataset class for loading annotated images and bounding boxes
class EagleEyesDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        with open(label_path, 'r') as f:
            self.data = json.load(f)['_default']
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = list(self.data.keys())

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        key = self.image_list[idx]
        item = self.data[key]
        filename = item['filename']
        image_path = os.path.join(self.image_dir, filename)

        # Load image based on file extension
        image = None
        try:
            if os.path.exists(image_path):
                if filename.endswith('.tiff') or filename.endswith('.ann.tiff'):
                    image = cv2.imread(image_path)
                    if image is None:
                        raise FileNotFoundError(f"Cannot load image: {image_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = Image.open(image_path).convert('RGB')
                    image = np.array(image)
            else:
                return None
        except Exception:
            return None

        # Convert bounding box from [y_center, x_center, w, h] to [x_min, y_min, x_max, y_max]
        boxes = []
        labels = []
        if 'annotations' in item['data'] and item['data']['annotations']:
            for ann in item['data']['annotations']:
                ijhw = ann['ijhw_box']
                y_center, x_center, w, h = map(int, ijhw)
                if w <= 0 or h <= 0:
                    continue
                x_min = x_center - w // 2
                y_min = y_center - h // 2
                x_max = x_center + w // 2
                y_max = y_center + h // 2
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(1)  # Label 1 for object class
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        image_tensor = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'filename': filename
        }

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, target, image

# Collate function to prepare batches
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    images, targets, orig_images = zip(*batch)
    return images, targets, orig_images

# Function to split dataset into train and test and copy images accordingly
def split_dataset(json_path, image_dir, train_image_dir, train_label_dir, test_image_dir, test_label_dir, max_images=None, train_ratio=0.8):
    for directory in [train_image_dir, train_label_dir, test_image_dir, test_label_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    train_label_path = os.path.join(train_label_dir, "train_labels.json")
    test_label_path = os.path.join(test_label_dir, "test_labels.json")

    if os.path.exists(train_label_path) and os.path.exists(test_label_path):
        return train_image_dir, train_label_path, test_image_dir, test_label_path

    with open(json_path, 'r') as f:
        data = json.load(f)['_default']

    keys = list(data.keys())
    total_image = len(keys)
    if max_images is None:
        max_images = total_image
    else:
        max_images = min(max_images, total_image)

    keys = keys[:max_images]
    random.shuffle(keys)
    train_size = int(train_ratio * len(keys))
    train_keys = keys[:train_size]
    test_keys = keys[train_size:]

    train_data = {"_default": {k: data[k] for k in train_keys}}
    test_data = {"_default": {k: data[k] for k in test_keys}}

    with open(train_label_path, 'w') as f:
        json.dump(train_data, f)
    with open(test_label_path, 'w') as f:
        json.dump(test_data, f)

    for key in train_keys:
        src = os.path.join(image_dir, data[key]['filename'])
        dst = os.path.join(train_image_dir, data[key]['filename'])
        if os.path.exists(src):
            shutil.copy(src, dst)

    for key in test_keys:
        src = os.path.join(image_dir, data[key]['filename'])
        dst = os.path.join(test_image_dir, data[key]['filename'])
        if os.path.exists(src):
            shutil.copy(src, dst)

    return train_image_dir, train_label_path, test_image_dir, test_label_path

# Function to draw and save visualized bounding boxes for annotations
def visualize_annotations(image_dir, label_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(label_path, 'r') as f:
        data = json.load(f)['_default']

    for key, item in data.items():
        filename = item['filename']
        image_path = os.path.join(image_dir, filename)

        image = None
        try:
            if os.path.exists(image_path):
                if filename.endswith('.tiff') or filename.endswith('.ann.tiff'):
                    image = cv2.imread(image_path)
                    if image is None:
                        raise FileNotFoundError(f"Cannot load image: {image_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.imread(image_path)
                    if image is None:
                        raise FileNotFoundError(f"Cannot load image: {image_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                continue
        except Exception:
            continue

        img_height, img_width = image.shape[:2]

        if 'annotations' in item['data'] and item['data']['annotations']:
            for ann in item['data']['annotations']:
                ijhw = ann['ijhw_box']
                y_center, x_center, w, h = map(int, ijhw)
                if w <= 0 or h <= 0:
                    continue

                x1 = max(0, min(x_center - w // 2, img_width))
                y1 = max(0, min(y_center - h // 2, img_height))
                x2 = max(0, min(x_center + w // 2, img_width))
                y2 = max(0, min(y_center + h // 2, img_height))

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        output_path = os.path.join(output_dir, f"viz_{filename.replace('.tiff', '.jpg')}")
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
