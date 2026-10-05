from data_loader import load_from_dataset_folders
from train_seg_and_class_trial_1 import DATA_PATH

# TODO: look at left/right most point and consider cropping the image

ready_sets = ["trial_set_01", "trial_set_02", "trial_set_03", "trial_set_04"]
samples = load_from_dataset_folders(DATA_PATH, included_sets=ready_sets)

left_most = 960
right_most = 960

left_lt_420 = 0
right_gt_1500 = 0
total = 0

for sample in samples:
    total += 1
    seg = sample["segmentation"][0]
    left = min(seg[::2])
    right = max(seg[::2])
    if left < 420:
        left_lt_420 += 1
    if right > 1500:
        right_gt_1500 += 1
    left_most = min(left_most, left)
    right_most = max(right_most, right)

print(left_most, right_most)
print(left_lt_420, right_gt_1500, total)
