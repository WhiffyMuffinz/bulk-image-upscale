import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
import torch
import shutil

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tile_cache import RunLocalTileCache
from tiled_processor import TiledBatchProcessor


class TestTiledProcessor(unittest.TestCase):
    def setUp(self):
        self.cache = RunLocalTileCache()

    def tearDown(self):
        self.cache.cleanup()

    def test_padding_handling(self):
        """Test that processor handles and crops padded output from model."""

        def mock_padded_pipeline(images, batch_size=None):
            # Return images that are larger than expected
            # Input 128 -> Scale 1 (for this test) -> Expected 128.
            # But let's return 136 (128+8).
            padded_images = []
            for img in images:
                w, h = img.size
                padded = Image.new(img.mode, (w + 8, h + 8), (0, 0, 0))
                padded.paste(img, (0, 0))
                padded_images.append(padded)
            return padded_images

        img = Image.new("RGB", (128, 128), color=(255, 0, 0))

        # Scale factor 1, so expected output 128. input pipeline gives 136.
        processor = TiledBatchProcessor(
            mock_padded_pipeline, self.cache, tile_size=128, overlap=0
        )
        processor.scale_factor = 1

        output_images = processor.process_batch([img])

        # Reconstruction shouldn't fail, and output should be 128x128 (input size)
        self.assertEqual(output_images[0].size, (128, 128))

    def test_deduplication(self):
        """Test that identical tiles are only processed once."""
        # Create a Mock Pipeline
        # It should return the input image as "upscaled" (pass-through) to simple verification
        # But we need to handle list input

        call_counter = MagicMock()

        def mock_pipeline(images, batch_size=None):
            call_counter(len(images))
            # Return same images (acting as Identity upscale)
            return images

        # Create a solid color image (256x256) -> 4 tiles of 128x128
        # All 4 tiles will be identical
        img = Image.new("RGB", (256, 256), color=(255, 0, 0))

        # Create a batch of 2 identical images
        batch = [img, img]

        # Initialize processor
        # Set overlap to 0 for simple math in this test
        processor = TiledBatchProcessor(
            mock_pipeline, self.cache, tile_size=128, overlap=0, batch_size=32
        )
        processor.scale_factor = 1  # Mock pipeline doesn't upscale

        # Run
        _ = processor.process_batch(batch)

        # Assertions
        # Total tiles = 4 per image * 2 images = 8 tiles.
        # Unique tiles = 1 (solid red).
        # Pipeline should have been called once (or multiple times if batching, but total items processed should be 1)

        total_processed = sum(args[0][0] for args in call_counter.call_args_list)
        print(f"Total tiles processed by pipeline: {total_processed}")

        self.assertEqual(total_processed, 1, "Should have only processed 1 unique tile")

    def test_reconstruction_seamless(self):
        """Test that an image is reconstructed correctly (using Identity pipeline)."""

        def mock_pipeline(images, batch_size=None):
            return images

        # Create a gradient image 256x256
        # If we use 0 overlap, reconstruction is trivial.
        # If we use overlap, we test blending.

        # Let's use numpy to create a deterministic gradient
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                arr[y, x] = [x % 255, y % 255, (x + y) % 255]

        original_img = Image.fromarray(arr)

        processor = TiledBatchProcessor(
            mock_pipeline, self.cache, tile_size=128, overlap=16, batch_size=32
        )
        processor.scale_factor = 1

        output_images = processor.process_batch([original_img])
        reconstructed_img = output_images[0]

        # Verify sizes
        self.assertEqual(reconstructed_img.size, original_img.size)

        # Verify content (allow small float error due to blending)
        # Convert to numpy
        orig_arr = np.array(original_img).astype(float)
        rec_arr = np.array(reconstructed_img).astype(float)

        diff = np.abs(orig_arr - rec_arr)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"Max pixel difference: {max_diff}")
        print(f"Mean pixel difference: {mean_diff}")

        # With consistent weighting, reconstruction of original signal might have slight error at edges if weighting isn't perfect unity sum,
        # but with the 'div by weight_acc' method it should be very close.
        self.assertLess(mean_diff, 1.0, "Reconstruction should be nearly identical")

    def test_cache_interop(self):
        """Test that cache actually stores data."""

        def mock_pipeline(images, batch_size=None):
            return images

        img = Image.new("RGB", (128, 128), color=(0, 255, 0))
        processor = TiledBatchProcessor(
            mock_pipeline, self.cache, tile_size=128, overlap=0
        )
        processor.scale_factor = 1

        processor.process_batch([img])

        # Verify DB has entries
        conn = self.cache._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM tiles")
        count = cursor.fetchone()[0]

        self.assertTrue(count > 0, "Cache should contain tiles")


if __name__ == "__main__":
    unittest.main()
