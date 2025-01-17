import os
import queue
from concurrent.futures import ThreadPoolExecutor
import itertools

import torch
import logging
from PIL import Image
from datasets import Dataset, IterableDataset
from tqdm import tqdm
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def producer(image_dir, image_queue, batch_size):
    """producer class for a shared queue"""
    dataset = load_image_dataset(image_dir, streaming=True)
    dataset_iterator = iter(dataset)

    for batch in batch_iterator(dataset_iterator, batch_size):
        image_queue.put(batch)

    # Signal the consumers to finish
    for _ in range(len(pipelines)):
        image_queue.put(None)


def consumer(pipeline, image_queue, output_dir):
    """consumer class for a shared queue"""
    while True:
        batch = image_queue.get()
        if batch is None:
            break
        process_images_on_device(batch, pipeline, output_dir)


def setup_pipelines(model_id="caidas/swin2SR-lightweight-x2-64"):
    """
    Set up and initialize image-to-image pipelines on available devices.

    This function initializes image-to-image pipelines using the specified model ID
    on all available GPU devices. If no GPUs are available, it initializes the pipeline
    on the CPU. It handles exceptions during the initialization process and logs errors.
    Each pipeline is configured to use the appropriate data type (float16 for GPU and float32 for CPU).

    Parameters:
    ----------
    model_id : str, optional
        The ID of the model to be used for the image-to-image pipeline.
        Defaults to "caidas/swin2SR-lightweight-x2-64".

    Returns:
    -------
    list
        A list of initialized pipeline objects, each associated with a device.
        If no devices are available or initialization fails for all devices, the list will be empty.

    Notes:
    -----
    - This function is used within the `process_images_with_parallel_devices` function to initialize
      pipelines for parallel image processing.
    - The function logs an informational message for each pipeline initialization attempt and an error
      message if an exception occurs.
    - The pipelines are configured to use float16 precision on GPU devices to reduce memory usage and
      potentially improve performance.
    - If no GPUs are available, the pipeline is initialized on the CPU using float32 precision.
    """
    pipelines = []
    devices = (
        [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else ["cpu"]
    )

    for device in devices:
        try:
            logging.info(f"Initializing pipeline on {device}")
            pipelines.append(
                pipeline(
                    "image-to-image",
                    model=model_id,
                    device=device,
                    torch_dtype=torch.float16 if "cuda" in device else None,
                )
            )
        except Exception as e:
            logging.error(f"Failed to initialize pipeline on {device}: {e}")

    return pipelines


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


def batch_iterator(iterable, batch_size):
    """
    Yield batches of a specified size from an iterable.

    This function takes an iterable and a batch size as input and yields
    batches of the specified size. If the number of elements in the iterable
    is not a perfect multiple of the batch size, the last batch will contain
    the remaining elements.

    Parameters:
    iterable (iterable): The iterable from which to yield batches.
    batch_size (int): The size of each batch. Must be a positive integer.

    Yields:
    list: A list containing elements from the iterable, up to `batch_size` elements.

    Example:
    >>> list(batch_iterator([1, 2, 3, 4, 5, 6, 7], 3))
    [[1, 2, 3], [4, 5, 6], [7]]
    """
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def process_images_on_device(device_index, subset, pipeline, batch_size, output_dir):
    """
    Process images on a specified device using an image-to-image pipeline and save the processed images.

    This function takes a subset of images, processes them in batches using the provided image-to-image
    pipeline, and saves the processed images to the specified output directory. It handles exceptions
    during the processing and saving of images and logs appropriate messages.

    Parameters:
    ----------
    device_index : int
        The index of the device on which the pipeline is running. This is mainly used for logging purposes.
    subset : iterable
        An iterable of image data, where each item is a dictionary containing the image and its path.
    pipeline : transformers.pipelines.Pipeline
        The image-to-image pipeline used to process the images.
    batch_size : int
        The size of each batch to process. Images are processed in batches to optimize memory usage.
    output_dir : str
        The directory where the processed images will be saved.

    Returns:
    -------
    None

    Notes:
    -----
    - The function ensures that the output directory exists before saving images.
    - It uses the `tqdm` library to provide a progress bar for batch processing.
    - The function logs informational messages at the start of the processing and debug messages upon
      successful saving of each processed image.
    - It handles exceptions during the loading and saving of images, logging error messages if any occur.
    - The function processes images in batches to avoid loading the entire dataset into memory at once.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Starting image processing pipeline")

    # Process in batches without materializing the full dataset
    for batch in tqdm(
        batch_iterator(subset, 50 * batch_size), desc=f"device: {pipeline.model.device}"
    ):
        # Extract images and paths for current batch
        batch_images = [item["image"] for item in batch]
        batch_paths = [item["path"] for item in batch]

        # Process batch
        try:
            outputs = pipeline(batch_images, batch_size=batch_size)

            # Save processed images
            for path, output in zip(batch_paths, outputs):
                output_path = os.path.join(output_dir, os.path.basename(path))
                try:
                    output.save(output_path)
                    logging.debug(f"Saved processed image to {output_path}")
                except Exception as e:
                    logging.error(f"Failed to save image {path}: {str(e)}")

        except Exception as e:
            logging.error(f"Failed to process batch: {str(e)}")
            continue


def process_images_with_parallel_devices(image_dir, pipelines, batch_sizes, output_dir):
    """
    Process images from a specified directory using multiple image-to-image pipelines in parallel.
    This function determines the optimal batch size for each pipeline, processes the images in batches,
    and saves the processed images to the specified output directory.

    Parameters:
    ----------
    image_dir : str
        The directory containing the input images to be processed.
    pipelines : list of transformers.pipelines.Pipeline
        A list of initialized image-to-image pipelines, each associated with a device.
        These pipelines will be used to process the images in parallel.
    batch_sizes : list of int
        A list of batch sizes corresponding to each pipeline. This parameter is currently
        not used directly in the function, but can be used to pass pre-determined batch sizes.
        If an empty list is provided, the function will determine the batch sizes internally.
    output_dir : str
        The directory where the processed images will be saved.

    Returns:
    -------
    None

    Notes:
    -----
    - The function loads an image from the dataset for testing batch sizes.
    - It determines the maximum batch size for each pipeline using the `find_max_batch_size` function.
    - The function uses a `ThreadPoolExecutor` to process images in parallel on all available devices.
    - The images are processed in batches to optimize memory usage, and the batch size is determined dynamically
      based on the available GPU memory.
    - The function handles exceptions during image processing and saving, logging appropriate messages.
    - The `batch_sizes` parameter is currently not used in the function and is provided for future use or flexibility.
    """
    # Load an image from the dataset for testing
    dataset = load_image_dataset(image_dir, streaming=True)
    if not dataset:
        logging.error("No images found in the directory")
        return

    # Get the first image for batch size testing
    dataset_iterator = iter(dataset)
    first_item = next(dataset_iterator)
    random_test_image = first_item["image"]

    # Create a new iterator that includes the first item
    dataset_iterator = itertools.chain([first_item], dataset_iterator)

    try:
        # Test batch size for each pipeline
        batch_sizes = [
            find_max_batch_size(pipe, random_test_image, 1, 10) for pipe in pipelines
        ]

        logging.info(f"Batch sizes found: {batch_sizes}")

        # Process images in parallel on all devices
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_images_on_device,
                    device_index=i,
                    subset=dataset_iterator,
                    pipeline=pipeline,
                    batch_size=batch_size,
                    output_dir=output_dir,
                )
                for i, (pipeline, batch_size) in enumerate(zip(pipelines, batch_sizes))
            ]

            for future in tqdm(futures, desc="Overall Progress"):
                future.result()
    except Exception as e:
        logging.error(f"An error occurred: {e}")


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


if __name__ == "__main__":
    model_id = "caidas/swin2SR-lightweight-x2-64"
    image_dir = "./input"
    output_dir = "./output"

    # Setup pipelines for all available devices
    pipelines = setup_pipelines(model_id=model_id)

    if not pipelines:
        logging.error("No pipelines configured. Exiting.")
        exit(1)

    process_images_with_parallel_devices(
        image_dir=image_dir, pipelines=pipelines, batch_sizes=[], output_dir=output_dir
    )
