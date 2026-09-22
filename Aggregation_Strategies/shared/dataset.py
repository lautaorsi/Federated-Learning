import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

NUM_CLASSES = 10
DIRICHLET_ALPHA = 0.5
RANDOM_SEED = 42

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Absolute, stable paths -- every simulated client process (each is a
# separate OS process under Flower's simulation) needs to agree on exactly
# where things live on disk.
DATA_ROOT = str(Path.home() / ".cache" / "pytorchexample_cifar10")
PARTITIONS_ROOT = Path.home() / ".cache" / "pytorchexample_partitions"
RAW_SPLIT_PATH = PARTITIONS_ROOT / "raw_split.pt"


def _atomic_torch_save(obj, path: Path):
    """Write via a temp file + rename so a half-written file is never read
    by a different process racing to use the same cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)  # atomic on both POSIX and Windows


# ---------------------------------------------------------------------------
# 1. Raw CIFAR10 load + stratified 60/10/30 split, cached to disk ONCE
#    regardless of num_partitions/alpha (every partitioning scheme starts
#    from the same split).
# ---------------------------------------------------------------------------


def _build_raw_split():
    raw_train = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True)
    raw_test = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)

    train_labels = torch.from_numpy(np.array(raw_train.targets)).long()
    test_labels = torch.from_numpy(np.array(raw_test.targets)).long()

    train_images = torch.from_numpy(raw_train.data).float() / 255.0
    train_images = train_images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    test_images = torch.from_numpy(raw_test.data).float() / 255.0
    test_images = test_images.permute(0, 3, 1, 2)

    images = torch.cat([train_images, test_images], dim=0)
    labels = torch.cat([train_labels, test_labels], dim=0)

    # 60% train, then split the remaining 40% into 10/30 (val/test)
    x_train, x_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.40, random_state=RANDOM_SEED, stratify=labels
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.75, random_state=RANDOM_SEED, stratify=y_temp
    )

    blob = {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    }
    _atomic_torch_save(blob, RAW_SPLIT_PATH)
    return blob


def _ensure_raw_split():
    """Expensive (downloads + splits 70k images) -- runs at most once ever,
    on whichever process gets there first. Every later call, in every
    process, just reads the small cached file back."""

    if RAW_SPLIT_PATH.exists():
        return torch.load(RAW_SPLIT_PATH, weights_only=True)
    
    return _build_raw_split()


# ---------------------------------------------------------------------------
# 2. Dirichlet label-skew client partitioning, ALSO cached to disk per
#    (num_partitions, alpha). Each client process only ever loads its own
#    small shard -- never the full dataset.
# ---------------------------------------------------------------------------


def _fractions_to_counts(fracs, n):
    """Convert fractional class shares into integer sample counts summing to n."""
    raw = [float(f) * n for f in fracs]
    counts = [int(x) for x in raw]
    remainder = n - sum(counts)
    fractional_parts = sorted(
        range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=True
    )
    for i in fractional_parts[:remainder]:
        counts[i] += 1
    assert sum(counts) == n
    return counts


def _partition_indices(labels: torch.Tensor, num_partitions: int, alpha: float):
    """For each class, draw a Dirichlet(alpha, ..., alpha) vector over
    `num_partitions` clients and split that class's indices accordingly.
    Returns one index array per client."""
    rng = np.random.default_rng(RANDOM_SEED)
    class_fracs = rng.dirichlet([alpha] * num_partitions, size=NUM_CLASSES)

    labels_np = labels.numpy()
    client_indices = [[] for _ in range(num_partitions)]
    for class_index in range(NUM_CLASSES):
        class_idx = np.where(labels_np == class_index)[0]
        rng.shuffle(class_idx)
        counts = _fractions_to_counts(class_fracs[class_index], len(class_idx))
        start = 0
        for i, count in enumerate(counts):
            client_indices[i].extend(class_idx[start:start + count].tolist())
            start += count
    return [np.array(idx) for idx in client_indices]


def _partition_dir(num_partitions: int, alpha: float) -> Path:
    return PARTITIONS_ROOT / f"n{num_partitions}_a{alpha}"


def _client_shard_path(num_partitions: int, alpha: float, partition_id: int, split: str) -> Path:
    return _partition_dir(num_partitions, alpha) / f"client_{partition_id}_{split}.pt"


def _partitions_cached(num_partitions: int, alpha: float) -> bool:
    return all(
        _client_shard_path(num_partitions, alpha, i, split).exists()
        for i in range(num_partitions)
        for split in ("train", "val")
    )


def _build_partitions(num_partitions: int, alpha: float):
    """Expensive: runs at most once per (num_partitions, alpha) combo, on
    whichever process gets there first."""
    raw = _ensure_raw_split()

    train_client_idx = _partition_indices(raw["y_train"], num_partitions, alpha)
    val_client_idx = _partition_indices(raw["y_val"], num_partitions, alpha)

    for i in range(num_partitions):
        _atomic_torch_save(
            {"images": raw["x_train"][train_client_idx[i]], "labels": raw["y_train"][train_client_idx[i]]},
            _client_shard_path(num_partitions, alpha, i, "train"),
        )
        _atomic_torch_save(
            {"images": raw["x_val"][val_client_idx[i]], "labels": raw["y_val"][val_client_idx[i]]},
            _client_shard_path(num_partitions, alpha, i, "val"),
        )


from filelock import FileLock

def _ensure_partitions(num_partitions: int, alpha: float):
    if _partitions_cached(num_partitions, alpha):
        return
    PARTITIONS_ROOT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(PARTITIONS_ROOT / f"n{num_partitions}_a{float(alpha)}.lock")):
        if not _partitions_cached(num_partitions, alpha):  # re-check after waiting
            _build_partitions(num_partitions, alpha)



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_client_dataloaders(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    alpha: float = DIRICHLET_ALPHA,
):
    """Return (trainloader, valloader) for this client's Dirichlet-skewed shard.

    Cheap after the first call anywhere (any process): just reads this one
    client's small cached shard off disk, never touches the full dataset.
    """
    _ensure_partitions(num_partitions, alpha)

    train_blob = torch.load(_client_shard_path(num_partitions, alpha, partition_id, "train"), weights_only=True)
    val_blob = torch.load(_client_shard_path(num_partitions, alpha, partition_id, "val"), weights_only=True)
    trainloader = DataLoader(
        TensorDataset(train_blob["images"], train_blob["labels"]), batch_size=batch_size, shuffle=True
    )
    valloader = DataLoader(TensorDataset(val_blob["images"], val_blob["labels"]), batch_size=batch_size)
    return trainloader, valloader


def get_centralized_dataloader(batch_size: int = 128):
    """Load the full (non-partitioned) test set for server-side evaluation."""
    raw = _ensure_raw_split()
    return DataLoader(TensorDataset(raw["x_test"], raw["y_test"]), batch_size=batch_size)
