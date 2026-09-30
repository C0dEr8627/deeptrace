"""Tests for DeepTrace DataLoader construction."""

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ml.data.dataloaders import calculate_class_weights, create_dataloaders
from ml.data.dataset import DeepTraceDataset


class DataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manifest = self.root / "frames.csv"
        rows = []
        labels_by_split = {
            "train": ["fake", "fake", "real"],
            "validation": ["fake", "real"],
            "test": ["fake", "real"],
        }
        for split, labels in labels_by_split.items():
            for index, label in enumerate(labels):
                frame_id = f"{split}_{index}"
                image_path = self.root / f"{frame_id}.jpg"
                Image.new("RGB", (32, 32), color=(index * 40, 80, 120)).save(image_path)
                rows.append({
                    "frame_id": frame_id,
                    "frame_path": str(image_path),
                    "video_sample_id": f"video_{frame_id}",
                    "label": label,
                    "split": split,
                })

        with self.manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_class_weights_use_inverse_training_frequency(self):
        dataset = DeepTraceDataset(self.manifest, "train")
        weights = calculate_class_weights(dataset)
        self.assertEqual(tuple(weights.shape), (2,))
        self.assertAlmostEqual(weights[0].item(), 0.75, places=5)
        self.assertAlmostEqual(weights[1].item(), 1.5, places=5)

    def test_create_dataloaders_returns_expected_splits(self):
        train, validation, test, weights = create_dataloaders(
            self.manifest, batch_size=2, num_workers=0
        )
        self.assertEqual(len(train.dataset), 3)
        self.assertEqual(len(validation.dataset), 2)
        self.assertEqual(len(test.dataset), 2)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)
        self.assertEqual(len(test), 1)
        self.assertEqual(tuple(weights.shape), (2,))

    def test_validation_and_test_loaders_do_not_shuffle(self):
        _, validation, test, _ = create_dataloaders(self.manifest, batch_size=2)
        self.assertFalse(validation.sampler.__class__.__name__ == "RandomSampler")
        self.assertFalse(test.sampler.__class__.__name__ == "RandomSampler")

    def test_invalid_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            create_dataloaders(self.manifest, batch_size=0)


if __name__ == "__main__":
    unittest.main()
