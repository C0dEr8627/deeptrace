"""Unit tests for the DeepTrace PyTorch frame dataset."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from ml.data.dataset import DeepTraceDataset, LABEL_MAP


class TestDeepTraceDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "frame.png"
        Image.new("RGB", (320, 180), color=(120, 80, 40)).save(self.image_path)
        self.manifest_path = self.root / "frames.csv"
        pd.DataFrame([
            {"frame_id": "fake-1#frame_01", "frame_path": str(self.image_path),
             "video_sample_id": "fake-1", "label": "fake", "split": "train"},
            {"frame_id": "real-1#frame_01", "frame_path": str(self.image_path),
             "video_sample_id": "real-1", "label": "real", "split": "validation"},
            {"frame_id": "real-2#frame_01", "frame_path": str(self.image_path),
             "video_sample_id": "real-2", "label": "real", "split": "test"},
        ]).to_csv(self.manifest_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_image_as_normalized_rgb_tensor(self):
        dataset = DeepTraceDataset(self.manifest_path, "train")
        image, label = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(label, LABEL_MAP["fake"])

    def test_respects_split_and_label(self):
        dataset = DeepTraceDataset(self.manifest_path, "validation")
        self.assertEqual(len(dataset), 1)
        _, label = dataset[0]
        self.assertEqual(label, LABEL_MAP["real"])

    def test_rejects_invalid_split(self):
        with self.assertRaises(ValueError):
            DeepTraceDataset(self.manifest_path, "testing")

    def test_rejects_missing_manifest(self):
        with self.assertRaises(FileNotFoundError):
            DeepTraceDataset(self.root / "missing.csv", "train")

    def test_rejects_invalid_label(self):
        data = pd.read_csv(self.manifest_path)
        data.loc[0, "label"] = "unknown"
        data.to_csv(self.manifest_path, index=False)
        with self.assertRaises(ValueError):
            DeepTraceDataset(self.manifest_path, "train")

    def test_rejects_duplicate_frame_ids(self):
        data = pd.read_csv(self.manifest_path)
        data.loc[1, "frame_id"] = data.loc[0, "frame_id"]
        data.to_csv(self.manifest_path, index=False)
        with self.assertRaises(ValueError):
            DeepTraceDataset(self.manifest_path, "train")


if __name__ == "__main__":
    unittest.main()
