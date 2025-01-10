import os
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



def setup_pipelines(model_id="caidas/swin2SR-lightweight-x2-64"):
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
    Find the maximum batch size that can fit in GPU memory using either linear or binary search.
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


def process_images_on_device(device_index, subset, pipeline, batch_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Filter out None values and prepare data
    valid_items = [item for item in subset if item is not None]
    if not valid_items:
        return

    images = [item["image"] for item in valid_items]
    paths = [item["path"] for item in valid_items]

    # Process all images
    for path, output in tqdm(zip(paths, pipeline(images, batch_size=batch_size)), desc=f"device: {pipeline.model.device}"):
        output_path = os.path.join(output_dir, os.path.basename(path))
        output.save(output_path)
        logging.info(f"Saved processed image to {output_path}")



def process_images_with_parallel_devices(image_dir, pipelines, batch_sizes, output_dir):
    # Load a random image from the dataset for testing
    # dataset = ImageDataset(image_dir)
    dataset = load_image_dataset(image_dir, streaming=True)
    if not dataset:
        logging.error("No images found in the directory")
        return

    dataset_iterator, peek_iterator = itertools.tee(iter(dataset))

    # Peek at the first item
    first_item = next(peek_iterator)
    random_test_image = first_item["image"]

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
        return Dataset.from_dict({"image": [x["image"] for x in data], "path": [x["path"] for x in data]})



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
