import random
from collections import defaultdict, Counter


def random_split_samples(
    samples,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)

    n = len(samples)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_samples = samples[:n_train]
    if test_ratio > 0:
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]
    else:
        val_samples = samples[n_train:]
        test_samples = []

    return train_samples, val_samples, test_samples


def stratified_random_split_samples(
    samples,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    by_class = defaultdict(list)

    for sample in samples:
        by_class[sample["risk_class"]].append(sample)

    train_samples = []
    val_samples = []
    test_samples = []

    rng = random.Random(seed)

    for risk_class, class_samples in by_class.items():
        class_samples = list(class_samples)
        rng.shuffle(class_samples)

        n = len(class_samples)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        train_samples.extend(class_samples[:n_train])
        if test_ratio > 0:
            val_samples.extend(class_samples[n_train:n_train + n_val])
            test_samples.extend(class_samples[n_train + n_val:])
        else:
            val_samples.extend(class_samples[n_train:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    return train_samples, val_samples, test_samples


def preset_set_split_samples(
    samples,
    train_sets,
    val_sets,
    test_sets=None,
):
    train_sets = set(train_sets)
    val_sets = set(val_sets)
    test_sets = set(test_sets) if test_sets is not None else set()

    overlap = train_sets & val_sets
    overlap |= train_sets & test_sets
    overlap |= val_sets & test_sets

    if len(overlap) > 0:
        raise ValueError(f"Some sets appear in multiple splits: {sorted(overlap)}")

    train_samples = []
    val_samples = []
    test_samples = []
    unused_samples = []

    for sample in samples:
        set_name = sample["set_name"]

        if set_name in train_sets:
            train_samples.append(sample)
        elif set_name in val_sets:
            val_samples.append(sample)
        elif set_name in test_sets:
            test_samples.append(sample)
        else:
            unused_samples.append(sample)

    return train_samples, val_samples, test_samples, unused_samples


def summarize_split(name, samples):
    class_counts = Counter(sample["risk_class"] for sample in samples)
    level_counts = Counter(sample["risk_level"] for sample in samples)
    set_counts = Counter(sample["set_name"] for sample in samples)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Number of samples: {len(samples)}")
    print(f"Risk class counts: {dict(sorted(class_counts.items()))}")
    print(f"Risk level counts: {dict(sorted(level_counts.items()))}")
    print(f"Number of sets: {len(set_counts)}")
    print(f"Sets: {dict(sorted(set_counts.items()))}")
    