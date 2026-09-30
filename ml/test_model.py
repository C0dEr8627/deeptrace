"""Unit tests for the DeepTrace EfficientNet-B0 model factory."""

import unittest

import torch
from torch import nn

from ml.model import build_model, get_device


class EfficientNetModelTests(unittest.TestCase):
    def test_model_has_two_output_classes(self):
        model = build_model(pretrained=False)
        self.assertIsInstance(model.classifier[1], nn.Linear)
        self.assertEqual(model.classifier[1].out_features, 2)

    def test_backbone_is_frozen_by_default(self):
        model = build_model(pretrained=False)
        self.assertTrue(all(not p.requires_grad for p in model.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))

    def test_backbone_can_be_unfrozen(self):
        model = build_model(pretrained=False, freeze_backbone=False)
        self.assertTrue(all(p.requires_grad for p in model.features.parameters()))

    def test_forward_output_shape(self):
        model = build_model(pretrained=False).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (1, 2))

    def test_device_is_torch_device(self):
        self.assertIsInstance(get_device(), torch.device)


if __name__ == "__main__":
    unittest.main()
