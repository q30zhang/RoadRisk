import os
import json
import cv2
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from tqdm import tqdm

from data_loader import load_from_dataset_folders, RoadRiskDataset
from data_split import random_split_samples, stratified_random_split_samples, \
    preset_set_split_samples, summarize_split


# ready_sets = ["trial_set_01", "trial_set_02", "trial_set_03", "trial_set_04"]
ready_sets = ["trial_set_01", "trial_set_02", "trial_set_03", "trial_set_04",
              "L1_set_03", "L2_set_02", "L4_set_02"]

MODEL_VERSION = 1
MODEL_NAMES = [
    "unet_resnet34_multitask",
    "deeplab_v3_with_coordconv",
    "unet_resnet34_with_coordconv",
    ]

VERSION = 1

# -----------------------------
# Config
# -----------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DATA_PATH = os.path.join(ROOT, "data/raw")
# LABEL_PATH = os.path.join(DATA_PATH, "labels_part1.json")
# IMAGE_DIR = os.path.join(ROOT, "data/seg_and_class_trial_1/img_part1")

n_versions = 1
for checkpoint_path in os.listdir(os.path.join(ROOT, "results", "checkpoints")):
    if checkpoint_path.startswith(f"best_{MODEL_NAMES[MODEL_VERSION-1]}") and checkpoint_path.endswith(".pth"):
        n_versions += 1
CHECKPOINT_PATH = os.path.join(ROOT, "results", "checkpoints", f"best_{MODEL_NAMES[MODEL_VERSION-1]}_v{n_versions}.pth")
MODEL_INFO_PATH = os.path.join(ROOT, "results", "checkpoints", f"best_{MODEL_NAMES[MODEL_VERSION-1]}_v{n_versions}.txt")

IMG_SIZE = 512
BATCH_SIZE = 4
NUM_EPOCHS = 20
LR = 1e-4
NUM_RISK_CLASSES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Utilities
# -----------------------------

# def polygon_to_mask(segmentation, height, width):
#     """
#     Convert COCO-style polygon segmentation to binary mask.

#     segmentation example:
#     [
#       [x1, y1, x2, y2, x3, y3, ...]
#     ]
#     """
#     mask = np.zeros((height, width), dtype=np.uint8)

#     for poly in segmentation:
#         pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
#         pts = np.round(pts).astype(np.int32)
#         cv2.fillPoly(mask, [pts], color=1)

#     return mask


# def load_coco_like_annotations(json_path):
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     images = {img["id"]: img for img in data["images"]}
#     annotations = data["annotations"]

#     samples = []

#     for ann in annotations:
#         image_id = ann["image_id"]
#         img_info = images[image_id]

#         file_name = img_info["file_name"]
#         height = img_info["height"]
#         width = img_info["width"]

#         risk_level = int(float(ann["attributes"]["risk_level"]))

#         # Convert risk labels 1,2,3,4 to class indices 0,1,2,3
#         risk_class = risk_level - 1

#         samples.append(
#             {
#                 "file_name": file_name,
#                 "height": height,
#                 "width": width,
#                 "segmentation": ann["segmentation"],
#                 "risk_class": risk_class,
#                 "risk_level": risk_level,
#             }
#         )

#     return samples


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# -----------------------------
# Dataset
# -----------------------------

# class RoadRiskDataset(Dataset):
#     def __init__(self, samples, image_dir, transform=None):
#         self.samples = samples
#         self.image_dir = image_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]

#         image_path = os.path.join(self.image_dir, sample["file_name"])

#         image = cv2.imread(image_path)
#         if image is None:
#             raise FileNotFoundError(f"Could not read image: {image_path}")

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         mask = polygon_to_mask(
#             sample["segmentation"],
#             sample["height"],
#             sample["width"],
#         )

#         risk_class = sample["risk_class"]

#         if self.transform is not None:
#             transformed = self.transform(image=image, mask=mask)
#             image = transformed["image"]
#             mask = transformed["mask"]

#         # Albumentations returns mask as H x W. Make it 1 x H x W.
#         mask = mask.float().unsqueeze(0)

#         risk_class = torch.tensor(risk_class, dtype=torch.long)

#         return image, mask, risk_class


# -----------------------------
# Model
# -----------------------------

class UNetResNet34MultiTask(nn.Module):
    """
    U-Net for ROI segmentation + image-level risk classification.

    Segmentation head:
      outputs 1-channel ROI logit map.

    Classification head:
      uses encoder's final feature map and predicts risk class.
    """

    def __init__(self, num_risk_classes=4):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )

        # For ResNet-34 encoder, final encoder feature has 512 channels.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_risk_classes),
        )

    def forward(self, x):
        # Use segmentation_models_pytorch encoder/decoder directly.
        features = self.unet.encoder(x)

        decoder_output = self.unet.decoder(features)
        seg_logits = self.unet.segmentation_head(decoder_output)

        deepest_feature = features[-1]
        risk_logits = self.classifier(deepest_feature)

        return seg_logits, risk_logits


# VERSION 2 (by Gemini)
class AddCoords(nn.Module):
    """
    Appends normalized X and Y coordinate channels to the input tensor.
    Input: (B, C, H, W) -> Output: (B, C+2, H, W)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, _, h, w = x.shape
        
        # Create normalized Y coordinates [-1, 1]
        y_coords = torch.linspace(-1, 1, steps=h, device=x.device)
        y_coords = y_coords.unsqueeze(1).expand(h, w).unsqueeze(0).expand(batch_size, 1, h, w)
        
        # Create normalized X coordinates [-1, 1]
        x_coords = torch.linspace(-1, 1, steps=w, device=x.device)
        x_coords = x_coords.unsqueeze(0).expand(h, w).unsqueeze(0).expand(batch_size, 1, h, w)
        
        return torch.cat([x, x_coords, y_coords], dim=1)


class CoordConv2dModel(nn.Module):
    """
    A simple CoordConv model for segmentation and classification.
    This is a placeholder for the actual model implementations below.
    """
    def __init__(self, num_risk_classes=4, Model=smp.Unet):
        super().__init__()
        
        self.coord_adder = AddCoords()
        
        # Note: in_channels is now 5 (RGB + X + Y)
        self.seg_model = Model(
            encoder_name="resnet34",
            encoder_weights=None, # Cannot use strict imagenet weights with 5 channels
            in_channels=5,
            classes=1,
            activation=None,
        )
        
        # Manually load ImageNet weights for the 3 RGB channels to speed up training
        self._load_pretrained_rgb()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Global feature (512) + ROI-masked feature (512) = 1024
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_risk_classes),
        )

    def _load_pretrained_rgb(self):
        """Loads ImageNet weights into the first 3 channels of the 5-channel conv1"""
        pretrained_encoder = smp.encoders.get_encoder("resnet34", in_channels=3, depth=5, weights="imagenet")
        
        with torch.no_grad():
            # Copy all matching layers
            for name, param in self.seg_model.encoder.named_parameters():
                if name in pretrained_encoder.state_dict():
                    if name == "conv1.weight":
                        # Copy the 3 RGB channels, leave the 2 coordinate channels randomly initialized
                        param.data[:, :3, :, :] = pretrained_encoder.state_dict()[name].data
                    else:
                        param.data.copy_(pretrained_encoder.state_dict()[name].data)

    def masked_average_pool(self, feature, mask_prob):
        mask_small = F.interpolate(mask_prob, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        weighted = feature * mask_small
        numerator = weighted.sum(dim=(2, 3))
        denominator = mask_small.sum(dim=(2, 3)).clamp_min(1e-6)
        return numerator / denominator

    def forward(self, x):
        # 1. Add coordinates
        x_coord = self.coord_adder(x)
        
        # 2. Extract features
        features = self.seg_model.encoder(x_coord)
        
        # 3. Decode segmentation
        try:
            decoder_output = self.seg_model.decoder(*features)
        except TypeError:
            decoder_output = self.seg_model.decoder(features)

        seg_logits = self.seg_model.segmentation_head(decoder_output)
        
        # 4. ROI-Aware Classification
        deepest_feature = features[-1]
        global_vec = self.global_pool(deepest_feature).flatten(1)
        
        mask_prob = torch.sigmoid(seg_logits)
        roi_vec = self.masked_average_pool(deepest_feature, mask_prob)
        
        cls_vec = torch.cat([global_vec, roi_vec], dim=1)
        risk_logits = self.classifier(cls_vec)
        
        return seg_logits, risk_logits


class DeepLabV3ResNet34ROIAware(CoordConv2dModel):
    def __init__(self, num_risk_classes=4):
        super().__init__(num_risk_classes=num_risk_classes, Model=smp.DeepLabV3Plus)


class UNetResNet34withCoordConv(CoordConv2dModel):
    def __init__(self, num_risk_classes=4):
        super().__init__(num_risk_classes=num_risk_classes, Model=smp.Unet)


# -----------------------------
# Training / validation
# -----------------------------

def dice_loss_with_logits(logits, targets, eps=1e-7):
    """
    logits: B x 1 x H x W
    targets: B x 1 x H x W
    """
    probs = torch.sigmoid(logits)

    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dims)
    cardinality = torch.sum(probs + targets, dims)

    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def compute_iou_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def train_one_epoch(model, loader, optimizer, seg_loss_fn, cls_loss_fn, lambda_cls=1.0):
    model.train()

    running_loss = 0.0
    running_iou = 0.0
    correct = 0
    total = 0

    for images, masks, risk_classes in tqdm(loader, desc="Train", leave=False):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        risk_classes = risk_classes.to(DEVICE)

        optimizer.zero_grad()

        seg_logits, risk_logits = model(images)

        loss_seg = seg_loss_fn(seg_logits, masks)
        loss_seg_dice = dice_loss_with_logits(seg_logits, masks)
        loss_seg = loss_seg + loss_seg_dice

        loss_cls = cls_loss_fn(risk_logits, risk_classes)

        loss = loss_seg + lambda_cls * loss_cls

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_iou += compute_iou_from_logits(seg_logits.detach(), masks) * images.size(0)

        preds = torch.argmax(risk_logits, dim=1)
        correct += (preds == risk_classes).sum().item()
        total += images.size(0)

    return {
        "loss": running_loss / total,
        "iou": running_iou / total,
        "risk_acc": correct / total,
    }


@torch.no_grad()
def validate(model, loader, seg_bce_loss, cls_loss_fn, lambda_cls=1.0):
    model.eval()

    running_loss = 0.0
    running_iou = 0.0
    correct = 0
    total = 0

    for images, masks, risk_classes in tqdm(loader, desc="Val", leave=False):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        risk_classes = risk_classes.to(DEVICE)

        seg_logits, risk_logits = model(images)

        loss_seg_bce = seg_bce_loss(seg_logits, masks)
        loss_seg_dice = dice_loss_with_logits(seg_logits, masks)
        loss_seg = loss_seg_bce + loss_seg_dice

        loss_cls = cls_loss_fn(risk_logits, risk_classes)

        loss = loss_seg + lambda_cls * loss_cls

        running_loss += loss.item() * images.size(0)
        running_iou += compute_iou_from_logits(seg_logits, masks) * images.size(0)

        preds = torch.argmax(risk_logits, dim=1)
        correct += (preds == risk_classes).sum().item()
        total += images.size(0)

    return {
        "loss": running_loss / total,
        "iou": running_iou / total,
        "risk_acc": correct / total,
    }


# -----------------------------
# Main
# -----------------------------

def main():
    # samples = load_coco_like_annotations(LABEL_PATH)

    # print(f"Loaded {len(samples)} samples")

    # risk_counts = Counter([s["risk_class"] for s in samples])
    # print("Risk class counts, zero-based:", risk_counts)
    # print("Risk level counts, original 1-4:", Counter([s["risk_level"] for s in samples]))

    if VERSION == 1:
        train_transform = A.Compose(
            [
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),

                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    elif VERSION == 2:
        train_transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            
            # Simulate different camera mounting positions and FOVs
            # Rotate is 0 because dashcams don't roll sideways
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5),
            
            # Lighting and weather robustness
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3), # Helps with glare/shadows
            
            # Simulate occlusions (dirt on windshield, heavy glare)
            A.CoarseDropout(max_holes=4, max_height=64, max_width=64, min_holes=1, fill_value=0, p=0.3),
            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    val_transform = A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )

    samples = load_from_dataset_folders(DATA_PATH, included_sets=ready_sets)

    # train_sets = []
    # val_sets = []
    # test_sets = []

    # train_samples, val_samples, test_samples, unused_samples = preset_set_split_samples(
    #     samples,
    #     train_sets=train_sets,
    #     val_sets=val_sets,
    #     test_sets=test_sets,
    # )

    train_samples, val_samples, test_samples = stratified_random_split_samples(
        samples,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        seed=42,
    )

    summarize_split("Train", train_samples)
    summarize_split("Validation", val_samples)
    summarize_split("Test", test_samples)

    train_dataset = RoadRiskDataset(train_samples, transform=train_transform)
    val_dataset = RoadRiskDataset(val_samples, transform=val_transform)
    test_dataset = RoadRiskDataset(test_samples, transform=val_transform)

    # # Simple random split.
    # # For a small dataset, you may later want stratified split by risk level.
    # n_total = len(samples)
    # n_train = int(0.8 * n_total)
    # n_val = n_total - n_train

    # generator = torch.Generator().manual_seed(42)
    # train_subset, val_subset = random_split(samples, [n_train, n_val], generator=generator)

    # train_samples = list(train_subset)
    # val_samples = list(val_subset)

    # train_dataset = RoadRiskDataset(train_samples, IMAGE_DIR, transform=train_transform)
    # val_dataset = RoadRiskDataset(val_samples, IMAGE_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if MODEL_VERSION == 1:
        model = UNetResNet34MultiTask(num_risk_classes=NUM_RISK_CLASSES).to(DEVICE)
    elif MODEL_VERSION == 2:
        model = DeepLabV3ResNet34ROIAware().to(DEVICE)
    elif MODEL_VERSION == 3:
        model = UNetResNet34withCoordConv(num_risk_classes=NUM_RISK_CLASSES).to(DEVICE)

    # Segmentation loss.
    seg_bce_loss = nn.BCEWithLogitsLoss()
    # Focal loss (suggested by Gemini)
    seg_focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    if VERSION == 1:
        seg_loss_fn = seg_bce_loss
    elif VERSION == 2:
        seg_loss_fn = seg_focal_loss

    # Weighted classification loss because your trial file appears imbalanced.
    train_risk_classes = [s["risk_class"] for s in train_samples]
    class_counts = Counter(train_risk_classes)

    weights = []
    for c in range(NUM_RISK_CLASSES):
        weights.append(1.0 / max(class_counts[c], 1))

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * NUM_RISK_CLASSES
    weights = weights.to(DEVICE)

    print("Classification weights:", weights.detach().cpu().numpy())

    cls_loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            seg_loss_fn,
            cls_loss_fn,
            lambda_cls=1.0,
        )

        val_metrics = validate(
            model,
            val_loader,
            seg_loss_fn,
            cls_loss_fn,
            lambda_cls=1.0,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f}, "
            f"train IoU {train_metrics['iou']:.4f}, "
            f"train risk acc {train_metrics['risk_acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f}, "
            f"val IoU {val_metrics['iou']:.4f}, "
            f"val risk acc {val_metrics['risk_acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                CHECKPOINT_PATH,
            )
            with open(MODEL_INFO_PATH, "w") as f:
                f.write(
                    f"Model: {MODEL_NAMES[MODEL_VERSION-1]}\n"
                    f"Version: {VERSION}\n"
                    f"Training sets: {ready_sets}\n"
                    f"Image size: {IMG_SIZE}\n"
                    f"Batch size: {BATCH_SIZE}\n"
                    f"Learning rate: {LR}\n"
                    f"Number of epochs: {NUM_EPOCHS}\n"
                    f"Segmentation loss: {'FocalLoss' if VERSION==2 else 'BCEWithLogitsLoss'}\n"
                    f"Classification loss: CrossEntropyLoss with weights {weights.detach().cpu().numpy()}\n"
                    f"Epoch: {epoch}\n"
                    f"Train loss: {train_metrics['loss']:.4f}\n"
                    f"Train IoU: {train_metrics['iou']:.4f}\n"
                    f"Train risk acc: {train_metrics['risk_acc']:.4f}\n"
                    f"Val loss: {best_val_loss:.4f}\n"
                    f"Val IoU: {val_metrics['iou']:.4f}\n"
                    f"Val risk acc: {val_metrics['risk_acc']:.4f}\n"
                )
            print("Saved best model")


if __name__ == "__main__":
    main()
