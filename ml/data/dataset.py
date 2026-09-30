"""PyTorch dataset for DeepTrace extracted video frames."""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_MAP = {"fake": 0, "real": 1}
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REQUIRED_COLUMNS = {"frame_id", "frame_path", "video_sample_id", "label", "split"}


def get_image_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Return deterministic ImageNet preprocessing for EfficientNet-B0."""
    if image_size < 1:
        raise ValueError("image_size must be positive")
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DeepTraceDataset(Dataset):
    """Load frames from a frame-level CSV while preserving video split labels."""

    def __init__(self, manifest_path: str | Path, split: str, transform=None):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")

        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        data = pd.read_csv(self.manifest_path, dtype=str, keep_default_na=False)
        missing = REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        if data["frame_id"].duplicated().any():
            raise ValueError("Duplicate frame_id values in frame manifest")

        data = data.loc[data["split"] == split].copy().reset_index(drop=True)
        if data.empty:
            raise ValueError(f"No frames found for split: {split}")

        data["label"] = data["label"].str.lower()
        invalid_labels = set(data["label"]) - set(LABEL_MAP)
        if invalid_labels:
            raise ValueError(f"Invalid labels: {sorted(invalid_labels)}")
        if data[["frame_id", "frame_path", "video_sample_id"]].eq("").any().any():
            raise ValueError("frame_id, frame_path, and video_sample_id cannot be empty")

        self.data = data
        self.split = split
        self.transform = transform if transform is not None else get_image_transform()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.data.iloc[index]
        image_path = Path(row["frame_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Frame image not found: {image_path}")

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, LABEL_MAP[row["label"]]
