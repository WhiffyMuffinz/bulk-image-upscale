"""
Tiled batch processor implementation.
Handles splitting images into tiles, deduplicating them against the cache,
running inference, and reconstructing the images.
"""

import hashlib
import logging
import math
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from tile_cache import RunLocalTileCache


class TiledBatchProcessor:
    def __init__(
        self,
        pipeline,
        cache: RunLocalTileCache,
        tile_size: int = 128,
        overlap: int = 16,
        batch_size: int = 32,
    ):
        self.pipeline = pipeline
        self.cache = cache
        self.tile_size = tile_size
        self.overlap = overlap
        # Internal batch size for the model inference (tiles are small, so this can be larger than image batch size)
        self.inference_batch_size = batch_size

        # Assume 2x upscale for now (standard for these models)
        # Ideally this would be dynamic, but compare_upscale.py also hardcoded it.
        # We can try to infer it from a dry run if needed, but 2x is safe for swin2SR-lightweight-x2-64
        self.scale_factor = 2

    def _hash_tile(self, tile: Image.Image) -> str:
        """Compute SHA256 hash of a tile's pixel data."""
        # Ensure consistent bytes
        return hashlib.sha256(tile.tobytes()).hexdigest()

    def process_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Process a batch of images using tiled upscaling and deduplication.
        """
        all_tile_requests = []  # List of (image_idx, tile_idx, tile_image, tile_hash)
        tile_metadata = []  # List of metadata for reconstruction per image

        # 1. Decompose all images into tiles
        for img_idx, image in enumerate(images):
            width, height = image.size
            step = self.tile_size - self.overlap

            tiles_x = max(1, math.ceil((width - self.overlap) / step))
            tiles_y = max(1, math.ceil((height - self.overlap) / step))

            img_metadata = {
                "width": width,
                "height": height,
                "tiles_x": tiles_x,
                "tiles_y": tiles_y,
                "step": step,
                "tile_info": [],  # List of (x, y, hash)
            }

            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    # Calculate coordinates (same logic as compare_upscale.py)
                    x_start = (
                        min(tx * step, width - self.tile_size)
                        if width > self.tile_size
                        else 0
                    )
                    y_start = (
                        min(ty * step, height - self.tile_size)
                        if height > self.tile_size
                        else 0
                    )
                    x_end = min(x_start + self.tile_size, width)
                    y_end = min(y_start + self.tile_size, height)

                    # Adjust start if we're at the edge
                    if x_end - x_start < self.tile_size and width >= self.tile_size:
                        x_start = max(0, x_end - self.tile_size)
                    if y_end - y_start < self.tile_size and height >= self.tile_size:
                        y_start = max(0, y_end - self.tile_size)

                    # Extract tile
                    tile = image.crop((x_start, y_start, x_end, y_end))

                    # Pad if needed
                    original_size = tile.size
                    if tile.width < self.tile_size or tile.height < self.tile_size:
                        padded = Image.new(
                            "RGB", (self.tile_size, self.tile_size), (0, 0, 0)
                        )
                        padded.paste(tile, (0, 0))
                        tile = padded

                    tile_hash = self._hash_tile(tile)

                    # Store request
                    all_tile_requests.append(
                        {
                            "image_idx": img_idx,
                            "tile_image": tile,
                            "hash": tile_hash,
                            "coords": (x_start, y_start, x_end, y_end),
                            "padded_size": original_size,
                        }
                    )

                    img_metadata["tile_info"].append(
                        {
                            "hash": tile_hash,
                            "coords": (x_start, y_start, x_end, y_end),
                            "original_size": original_size,
                        }
                    )

            tile_metadata.append(img_metadata)

        # 2. Check Cache
        unique_hashes = set(req["hash"] for req in all_tile_requests)
        cached_tiles = self.cache.get_batch(list(unique_hashes))

        missing_hashes = unique_hashes - set(cached_tiles.keys())

        # 3. Infer missing tiles
        if missing_hashes:
            logging.info(
                f"Batch deduplication: {len(unique_hashes) - len(missing_hashes)}/{len(unique_hashes)} unique tiles found in cache"
            )

            # Group missing tiles by hash to process one instance of each
            tiles_to_process = {}  # hash -> image
            for req in all_tile_requests:
                if (
                    req["hash"] in missing_hashes
                    and req["hash"] not in tiles_to_process
                ):
                    tiles_to_process[req["hash"]] = req["tile_image"]

            # Process in chunks
            process_list = list(tiles_to_process.items())  # List of (hash, image)

            new_results = {}
            for i in range(0, len(process_list), self.inference_batch_size):
                chunk = process_list[i : i + self.inference_batch_size]
                chunk_hashes = [item[0] for item in chunk]
                chunk_images = [item[1] for item in chunk]

                try:
                    # Run inference
                    outputs = self.pipeline(chunk_images, batch_size=len(chunk_images))

                    # Move to CPU/Numpy and serialize
                    for h, output_img in zip(chunk_hashes, outputs):
                        # Handle potential output padding from model
                        expected_dim = self.tile_size * self.scale_factor

                        if (
                            output_img.width > expected_dim
                            or output_img.height > expected_dim
                        ):
                            output_img = output_img.crop(
                                (0, 0, expected_dim, expected_dim)
                            )

                        # Convert to bytes for storage
                        # We store as raw bytes of the 8-bit RGB array to be space efficient
                        # Alternatively could store as PNG bytes, but raw numpy is faster to load
                        arr = np.array(output_img).astype(np.uint8)
                        new_results[h] = arr.tobytes()

                except Exception as e:
                    logging.error(f"Inference error on tile batch: {e}")
                    raise e

            # Save new results to cache
            self.cache.put_batch(new_results)

            # Update local cache map
            cached_tiles.update(new_results)

        # 4. Reconstruct Images
        output_images = []

        for img_meta in tile_metadata:
            out_w = img_meta["width"] * self.scale_factor
            out_h = img_meta["height"] * self.scale_factor

            # Accumulators
            output_acc = np.zeros((out_h, out_w, 3), dtype=np.float64)
            weight_acc = np.zeros((out_h, out_w), dtype=np.float64)

            for tile_info in img_meta["tile_info"]:
                h = tile_info["hash"]
                if h not in cached_tiles:
                    logging.error(f"Missing tile hash {h} during reconstruction")
                    continue

                # Retrieve and deserialize
                tile_bytes = cached_tiles[h]
                # Reconstruct numpy array
                # Upscaled tile size
                ts = self.tile_size * self.scale_factor
                tile_arr = (
                    np.frombuffer(tile_bytes, dtype=np.uint8)
                    .reshape((ts, ts, 3))
                    .astype(np.float64)
                )

                # Crop if it was padded
                orig_w, orig_h = tile_info["original_size"]
                actual_w = orig_w * self.scale_factor
                actual_h = orig_h * self.scale_factor

                if actual_w < ts or actual_h < ts:
                    tile_arr = tile_arr[:actual_h, :actual_w, :]

                # Calculate placement
                x_start, y_start, x_end, y_end = tile_info["coords"]
                out_x_start = x_start * self.scale_factor
                out_y_start = y_start * self.scale_factor
                out_x_end = out_x_start + actual_w
                out_y_end = out_y_start + actual_h

                # Weight mask (calculated on the fly per tile size to handle edge cases correctly)
                weight = np.ones((actual_h, actual_w), dtype=np.float64)
                feather = self.overlap * self.scale_factor

                # Feathering logic (same as compare_upscale.py)
                # Note: This is slightly simplified; strictly we should know if it's an edge tile
                # to decide whether to feather. compare_upscale processes loop indices.
                # Here we reconstruct based on dimensions.
                # Ideally, we should only feather 'internal' edges.
                # But for simplicity, if we follow the extract logic, the tiles overlap everywhere except image edges.
                # Let's apply standard feathering logic but be careful at image boundaries.

                # Check edges against image boundaries
                is_left_edge = out_x_start == 0
                is_right_edge = out_x_end == out_w
                is_top_edge = out_y_start == 0
                is_bottom_edge = out_y_end == out_h

                # Feather Top
                if not is_top_edge:
                    for i in range(min(feather, actual_h // 2)):
                        factor = (i + 1) / feather
                        weight[i, :] *= factor

                # Feather Bottom
                if not is_bottom_edge:
                    for i in range(min(feather, actual_h // 2)):
                        factor = (i + 1) / feather
                        weight[actual_h - 1 - i, :] *= factor

                # Feather Left
                if not is_left_edge:
                    for i in range(min(feather, actual_w // 2)):
                        factor = (i + 1) / feather
                        weight[:, i] *= factor

                # Feather Right
                if not is_right_edge:
                    for i in range(min(feather, actual_w // 2)):
                        factor = (i + 1) / feather
                        weight[:, actual_w - 1 - i] *= factor

                # Accumulate
                output_acc[out_y_start:out_y_end, out_x_start:out_x_end] += (
                    tile_arr * weight[:, :, np.newaxis]
                )
                weight_acc[out_y_start:out_y_end, out_x_start:out_x_end] += weight

            # Normalize
            weight_acc = np.maximum(weight_acc, 1e-8)
            output_acc /= weight_acc[:, :, np.newaxis]

            # Finalize
            output_acc = np.clip(output_acc, 0, 255).astype(np.uint8)
            output_images.append(Image.fromarray(output_acc))

        return output_images
