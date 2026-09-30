"""DataLoader construction and class weighting for DeepTrace training."""

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ml.data.dataset import DeepTraceDataset, LABEL_MAP


def calculate_class_weights(dataset: DeepTraceDataset) -> Tensor:
    """Return inverse-frequency weights in fake/real label order.

    Weights are calculated from the supplied dataset only. Pass the training
    dataset here; validation and test labels must not influence training.
    The weights are normalized to have a mean of 1.
    """
    labels = [LABEL_MAP[label] for label in dataset.data["label"].tolist()]
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=len(LABEL_MAP))
    if torch.any(counts == 0):
        raise ValueError("Training dataset must contain both fake and real samples")

    weights = counts.sum().float() / (len(counts) * counts.float())
    return weights


def create_dataloaders(
    manifest_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, Tensor]:
    """Create train/validation/test loaders and weights from train labels."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    train_dataset = DeepTraceDataset(manifest_path, "train")
    validation_dataset = DeepTraceDataset(manifest_path, "validation")
    test_dataset = DeepTraceDataset(manifest_path, "test")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    class_weights = calculate_class_weights(train_dataset)
    return train_loader, validation_loader, test_loader, class_weights
