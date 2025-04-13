import os
import logging

import torch
from PIL import Image
from datasets import Dataset, IterableDataset


def find_max_batch_size(pipe, test_image, min_size=0, max_size=100, method="binary"):
    """
    Determine the maximum batch size that can fit in GPU memory without causing an out-of-memory error.
    This function uses either a linear or binary search method to find the optimal batch size for
    the given image-to-image pipeline.

    Parameters:
    ----------
    pipe : transformers.pipelines.Pipeline
        The image-to-image pipeline that processes the batch of images.
        This pipeline should be initialized with a model and a specific device.
    test_image : PIL.Image.Image
        A single test image to be used for determining the maximum batch size.
        This image will be replicated to form batches of varying sizes.
    min_size : int, optional
        The minimum batch size to test. Default is 0.
        The function will not test batch sizes smaller than this value.
    max_size : int, optional
        The maximum batch size to test. Default is 100.
        The function will not test batch sizes larger than this value.
    method : str, optional
        The search method to use for finding the maximum batch size.
        Options are "linear" or "binary". Default is "binary".

    Returns:
    -------
    int
        The maximum batch size that can fit in GPU memory without causing an out-of-memory error.
        If no valid batch size is found within the specified range, it returns the `min_size`.

    Raises:
    ------
    RuntimeError
        If an error other than an out-of-memory error occurs during testing.

    Notes:
    -----
    - This function is used within the `process_images_with_parallel_devices` function to determine
      the optimal batch size for each pipeline before processing the images in parallel.
    - The function assumes that the `pipe` function handles batching internally and can process a list of images.
    - The function logs a warning if the device is set to CPU, as batching on CPU is generally not recommended.
    - The function uses binary search by default, which is more efficient than linear search for large ranges.
    - If the `pipe` function raises a `RuntimeError` that contains "out of memory" in its message, the function
      clears the GPU cache and continues searching.
    - The `test_image` is obtained from the first image in the dataset loaded by `load_image_dataset`.
    - The determined batch sizes are used in the `process_images_on_device` function to process images in batches.
    """

    def test_batch_size(size):
        try:
            # Create a batch of the same test image
            batch = [test_image] * size
            # Let pipeline handle the batching internally
            pipe(batch, batch_size=size)
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            raise

    # do not batch on cpu
    if pipe.model.device == torch.device("cpu"):
        logging.warning("Using cpu. You probably don't want this.")
        return 1

    if method == "linear":
        batch_size = max_size
        while batch_size > min_size:
            if test_batch_size(batch_size):
                return batch_size
            batch_size -= 1
        return min_size

    else:  # binary search
        left = min_size
        right = max_size
        best_size = min_size

        while left <= right:
            mid = (left + right) // 2
            if test_batch_size(mid):
                best_size = mid
                left = mid + 1
            else:
                right = mid - 1

        return best_size


def load_image_dataset(image_dir, streaming=False):
    """
    Load images from a specified directory into a dataset, either as a streaming dataset or a fully loaded dataset.

    This function scans the specified directory for image files (with extensions .png, .jpg, .jpeg, .bmp),
    loads them into memory, and returns a dataset object. The dataset can be returned as a streaming dataset
    (using `IterableDataset`) or as a fully loaded dataset (using `Dataset`), depending on the `streaming` parameter.

    Parameters:
    ----------
    image_dir : str
        The directory containing the image files to be loaded.
    streaming : bool, optional
        If `True`, the function returns an `IterableDataset` that streams images one by one.
        If `False`, the function loads all images into memory and returns a `Dataset`.
        Default is `False`.

    Returns:
    -------
    dataset : Dataset or IterableDataset
        If `streaming` is `True`, returns an `IterableDataset` for streaming images.
        If `streaming` is `False`, returns a `Dataset` containing all images loaded into memory.

    Raises:
    ------
    FileNotFoundError
        If the specified `image_dir` does not exist or is not a directory.

    Notes:
    -----
    - The function converts each loaded image to RGB format to ensure consistency.
    - Errors encountered while loading individual images are caught and logged, but do not halt the loading process.
    - The function is used in the `process_images_with_parallel_devices` function to load images for processing.
    - In streaming mode, the dataset is generated on-the-fly, which is memory efficient for large datasets.
    - In non-streaming mode, all images are loaded into memory at once, which can be faster for small to medium-sized datasets.
    """

    def image_generator():
        for fname in os.listdir(image_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                img_path = os.path.join(image_dir, fname)
                try:
                    image = Image.open(img_path).convert("RGB")
                    yield {"image": image, "path": img_path}
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    if streaming:
        # Return an IterableDataset for streaming
        return IterableDataset.from_generator(image_generator)
    else:
        # Load all data into memory for non-streaming mode
        data = list(image_generator())
        return Dataset.from_dict(
            {"image": [x["image"] for x in data], "path": [x["path"] for x in data]}
        )
