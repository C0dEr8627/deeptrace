"""Unit tests for deterministic frame selection and path-safe IDs."""

import unittest

from extract_frames import frame_indices, safe_id


class TestFrameExtractionHelpers(unittest.TestCase):
    def test_evenly_spaced_indices_include_endpoints(self):
        self.assertEqual(frame_indices(101, 5), [0, 25, 50, 75, 100])

    def test_short_video_uses_available_unique_frames(self):
        self.assertEqual(frame_indices(3, 5), [0, 1, 2])

    def test_empty_video_has_no_indices(self):
        self.assertEqual(frame_indices(0, 5), [])

    def test_non_positive_requested_count_is_rejected(self):
        with self.assertRaises(ValueError):
            frame_indices(10, 0)

    def test_sample_id_is_filesystem_safe(self):
        self.assertEqual(safe_id("celebdf_v2/Celeb-real/id13_0000"), "celebdf_v2__Celeb-real__id13_0000")


if __name__ == "__main__":
    unittest.main()
