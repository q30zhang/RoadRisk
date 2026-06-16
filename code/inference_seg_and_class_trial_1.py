import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import json

# Reuse the model class from above.
from train_seg_and_class_trial_1 import UNetResNet34MultiTask, DeepLabV3ResNet34ROIAware
from train_seg_and_class_trial_1 import VERSION, MODEL_NAMES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
IMG_PATH = os.path.join(ROOT, "data", "seg_and_class_trial_1", "img_part2")
CHECKPOINT_PATH = os.path.join(ROOT, "results", "checkpoints", f"best_{MODEL_NAMES[VERSION-1]}.pth")
RESULTS_LABEL_PATH = os.path.join(ROOT, "results", "results_seg_and_class_trial_1.json")
RESULTS_IMG_PATH = os.path.join(ROOT, "results", "img", "seg_and_class_trial_1", f"v{VERSION}")

if VERSION == 1:
    model = UNetResNet34MultiTask(num_risk_classes=4).to(DEVICE)
elif VERSION == 2:
    model =DeepLabV3ResNet34ROIAware(num_risk_classes=4).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]
)


def inference(image_name):
    image_path = os.path.join(IMG_PATH, image_name)
    original_bgr = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    h0, w0 = original_rgb.shape[:2]

    x = transform(image=original_rgb)["image"]
    x = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seg_logits, risk_logits = model(x)

    seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
    seg_mask_small = (seg_prob > 0.5).astype(np.uint8)

    # Resize predicted mask back to original image size.
    seg_mask = cv2.resize(seg_mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST)

    risk_class = torch.argmax(risk_logits, dim=1).item()
    risk_level = risk_class + 1

    # print("Predicted risk level:", risk_level)

    # Optional visualization overlay.
    overlay = original_rgb.copy()
    overlay[seg_mask == 1] = (
        0.5 * overlay[seg_mask == 1] + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(RESULTS_IMG_PATH, f"overlay_pred_{image_name}"), overlay_bgr)
    return risk_level


if __name__ == '__main__':
    if not os.path.exists(RESULTS_IMG_PATH):
        os.mkdir(RESULTS_IMG_PATH)
    image_names = filter(lambda x: x.endswith(".jpg") or x.endswith(".png"), os.listdir(IMG_PATH))
    results = {}
    for image_name in image_names:
        risk_level = inference(image_name)
        results.update({image_name: risk_level})
    with open(RESULTS_LABEL_PATH, 'w') as f:
        json.dump(results, f)
