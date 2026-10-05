import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path


def polygon_to_mask(segmentation, height, width):
    """
    Convert COCO-style polygon segmentation to binary mask.

    segmentation example:
    [
      [x1, y1, x2, y2, x3, y3, ...]
    ]
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for poly in segmentation:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], color=1)

    return mask


def load_coco_like_annotations(json_path, image_dir):
    json_path = Path(json_path)
    image_dir = Path(image_dir)

    with open(json_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    annotations = data["annotations"]

    samples = []

    for ann in annotations:
        if ann["attributes"]["unclear"]:
            continue
        if ann["attributes"].get("night_time") and ann["attributes"]["night_time"]:
            continue
        image_id = ann["image_id"]
        img_info = images[image_id]

        file_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]

        risk_level = int(float(ann["attributes"]["risk_level"]))

        # Convert risk labels 1,2,3,4 to class indices 0,1,2,3.
        risk_class = risk_level - 1

        image_path = image_dir / file_name

        samples.append(
            {
                "file_name": file_name,
                "image_path": str(image_path),
                "height": height,
                "width": width,
                "segmentation": ann["segmentation"],
                "risk_class": risk_class,
                "risk_level": risk_level,
                "source_json": str(json_path),
                "source_image_dir": str(image_dir),
            }
        )

    return samples


def load_from_dataset_folders(data_root, excluded_sets=None, included_sets=None):
    data_root = Path(data_root)

    all_samples = []

    dataset_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])

    for dataset_dir in dataset_dirs:
        if excluded_sets and dataset_dir.name in excluded_sets:
            continue
        if included_sets and dataset_dir.name not in included_sets:
            continue
        json_path = dataset_dir / "instances_default.json"
        image_dir = dataset_dir / "images"

        if not json_path.exists():
            raise FileNotFoundError(f"Missing instances_default.json in: {dataset_dir}")

        if not image_dir.exists():
            raise FileNotFoundError(f"Missing images folder in: {dataset_dir}")

        samples = load_coco_like_annotations(
            json_path=json_path,
            image_dir=image_dir,
        )

        # Important: store which set this sample came from.
        for sample in samples:
            sample["set_name"] = dataset_dir.name

        all_samples.extend(samples)

    return all_samples


class RoadRiskDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = sample["image_path"]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = polygon_to_mask(
            sample["segmentation"],
            sample["height"],
            sample["width"],
        )

        risk_class = sample["risk_class"]

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Albumentations returns mask as H x W. Make it 1 x H x W.
        mask = mask.float().unsqueeze(0)

        risk_class = torch.tensor(risk_class, dtype=torch.long)

        return image, mask, risk_class
