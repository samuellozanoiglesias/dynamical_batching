"""
generate_dataset_gridCNN.py

CIFAR-10-only, grid-search-optimised version of generate_dataset.py.

What was removed vs. the original:
  * every other dataset (MNIST, CIFAR-100, Tiny-ImageNet, spiral, blobs, ...)
  * metaclass remapping, dimensionality reduction (PCA/UMAP/t-SNE/...)
  * image-grid / class-distribution visualisations (matplotlib, per run!)

What was added:
  * A one-time uint8 cache (`cifar10_uint8.npz`) next to the extracted
    archive. Loading it takes a fraction of a second instead of unpickling
    the 6 python batches for every single run. It is written atomically
    (tmp file + rename), so many concurrent runs can never read a half
    written cache.
  * Images are returned as **uint8** (N, 3, 32, 32). Normalisation with the
    exact same mean/std as the original loader is done on the GPU inside the
    training loop (see training_runner_gridCNN.py), which keeps each process'
    GPU footprint at ~150 MB for the training set instead of ~600 MB.

Pre-build the cache once before launching the grid (the sbatch script does it):
    python generate_dataset_gridCNN.py --data_dir /path/to/data
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CACHE_NAME = "cifar10_uint8.npz"


@dataclass
class CIFAR10Data:
    train_images_u8: np.ndarray  # (50000, 3, 32, 32) uint8
    train_targets: np.ndarray    # (50000,) int64
    test_images_u8: np.ndarray   # (10000, 3, 32, 32) uint8
    test_targets: np.ndarray     # (10000,) int64
    class_to_indices: Dict[int, np.ndarray]
    num_classes: int = 10
    input_shape: tuple = (3, 32, 32)


def build_class_index_map(labels: np.ndarray, num_classes: int) -> Dict[int, np.ndarray]:
    labels_np = np.asarray(labels, dtype=np.int64)
    out: Dict[int, np.ndarray] = {}
    for class_id in range(num_classes):
        idx = np.where(labels_np == class_id)[0].astype(np.int64)
        if idx.size == 0:
            raise ValueError(f"Class {class_id} has no samples in training data.")
        out[class_id] = idx
    return out


def _unpickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _ensure_extracted(data_dir: Path) -> Path:
    extracted = data_dir / "cifar-10-batches-py"
    if extracted.exists():
        return extracted
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / Path(CIFAR10_URL).name
    if archive.exists() and not tarfile.is_tarfile(archive):
        archive.unlink()
    if not archive.exists():
        tmp = archive.with_suffix(archive.suffix + f".part{os.getpid()}")
        req = urllib.request.Request(CIFAR10_URL, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req) as resp, open(tmp, "wb") as dst:
                shutil.copyfileobj(resp, dst)
            os.replace(tmp, archive)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(data_dir)
    return extracted


def _build_cache(data_dir: Path, cache_path: Path) -> None:
    extracted = _ensure_extracted(data_dir)
    train_batches = [_unpickle(extracted / f"data_batch_{i}") for i in range(1, 6)]
    x_train = np.concatenate([b[b"data"] for b in train_batches], axis=0).reshape(-1, 3, 32, 32)
    y_train = np.concatenate([np.asarray(b[b"labels"]) for b in train_batches], axis=0)
    test = _unpickle(extracted / "test_batch")
    x_test = np.asarray(test[b"data"]).reshape(-1, 3, 32, 32)
    y_test = np.asarray(test[b"labels"])

    tmp = cache_path.with_name(cache_path.name + f".tmp{os.getpid()}.npz")
    np.savez(
        tmp,
        x_train=np.ascontiguousarray(x_train, dtype=np.uint8),
        y_train=y_train.astype(np.int64),
        x_test=np.ascontiguousarray(x_test, dtype=np.uint8),
        y_test=y_test.astype(np.int64),
    )
    os.replace(tmp, cache_path)  # atomic: concurrent readers see old-or-new, never partial


def load_cifar10(data_dir: str | Path) -> CIFAR10Data:
    data_dir = Path(str(data_dir)).resolve()
    cache_path = data_dir / CACHE_NAME
    if not cache_path.exists():
        _build_cache(data_dir, cache_path)
    with np.load(cache_path) as z:
        x_train, y_train = z["x_train"], z["y_train"]
        x_test, y_test = z["x_test"], z["y_test"]
    return CIFAR10Data(
        train_images_u8=x_train,
        train_targets=y_train,
        test_images_u8=x_test,
        test_targets=y_test,
        class_to_indices=build_class_index_map(y_train, 10),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-build the CIFAR-10 uint8 cache.")
    parser.add_argument("--data_dir", default="./data")
    args = parser.parse_args()
    d = load_cifar10(args.data_dir)
    print(
        f"CIFAR-10 cache ready at {Path(args.data_dir).resolve() / CACHE_NAME}: "
        f"train {d.train_images_u8.shape}, test {d.test_images_u8.shape}"
    )
