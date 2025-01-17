import os
import queue
from concurrent.futures import ThreadPoolExecutor
import itertools
import threading

import torch
import logging
from PIL import Image
from datasets import Dataset, IterableDataset
from tqdm import tqdm
from transformers import pipeline

from upscale_parallel_batches import (
    load_image_dataset,
    find_max_batch_size,
    batch_iterator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def producer(image_dir, image_queue):
    """
    Produces batches of images and puts them in the shared queue.

    Parameters:
    ----------
    image_dir : str
        Directory containing input images
    image_queue : queue.Queue
        Shared queue for image batches
    batch_size : int
        Size of image batches to create
    """
    dataset = load_image_dataset(image_dir, streaming=True)
    dataset_iterator = iter(dataset)

    try:
        for image in dataset_iterator:
            image_queue.put(image)
    except Exception as e:
        logging.error(f"Producer error: {e}")
    finally:
        # Signal the consumers to finish by adding None for each consumer
        for _ in range(torch.cuda.device_count()):
            image_queue.put(None)


def consumer(pipeline, image_queue, output_dir, batch_size):
    """
    Consumes image batches from the queue and processes them using the given pipeline.

    Parameters:
    ----------
    pipeline : transformers.Pipeline
        Pipeline instance for processing images
    image_queue : queue.Queue
        Shared queue containing image batches
    output_dir : str
        Directory to save processed images
    """
    device_index = (
        pipeline.model.device.index if hasattr(pipeline.model.device, "index") else -1
    )
    device_name = (
        torch.cuda.get_device_name(device_index) if device_index != -1 else "CPU"
    )

    while True:
        try:
            flag = False
            batch = []

            for _ in range(batch_size):
                image = image_queue.get(timeout=60)
                if image is None:
                    logging.info(f"Consumer on {device_name} received shutdown signal")
                    flag = True
                    break
                batch.append(image_queue.get(timeout=60))  # 60 second timeout

            # Process the batch
            batch_images = [item["image"] for item in batch]
            batch_paths = [item["path"] for item in batch]

            logging.info(
                f"Consumer on {device_name} received batch: len:{len(batch_images)} "
            )

            outputs = pipeline(batch_images, batch_size=batch_size)

            # Save processed images
            for path, output in zip(batch_paths, outputs):
                output_path = os.path.join(output_dir, os.path.basename(path))
                try:
                    output.save(output_path)
                    logging.debug(f"Saved processed image to {output_path}")
                except Exception as e:
                    logging.error(f"Failed to save image {path}: {str(e)}")
            if flag == True:
                break
        except queue.Empty:
            logging.warning(f"Queue timeout on {device_name}")
            continue
        except Exception as e:
            logging.error(f"Consumer error on {device_name}: {e}")
            continue
        finally:
            image_queue.task_done()


def process_images_with_queue(
    image_dir, output_dir, model_id="caidas/swin2SR-lightweight-x2-64", queue_size=100
):
    """
    Process images using multiple GPUs with a shared queue system.

    Parameters:
    ----------
    image_dir : str
        Input directory containing images
    output_dir : str
        Output directory for processed images
    model_id : str
        Model identifier for the pipeline
    queue_size : int
        Maximum size of the shared queue
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create shared queue
    image_queue = queue.Queue(maxsize=queue_size)

    # Initialize pipelines on all available GPUs
    pipelines = []
    for i in range(torch.cuda.device_count()):
        device = f"cuda:{i}"
        try:
            pipe = pipeline(
                "image-to-image",
                model=model_id,
                device=device,
                torch_dtype=torch.float16,
            )
            pipelines.append(pipe)
            logging.info(f"Initialized pipeline on {device}")
        except Exception as e:
            logging.error(f"Failed to initialize pipeline on {device}: {e}")

    if not pipelines:
        raise RuntimeError("No pipelines could be initialized")

    # Find optimal batch size using the first pipeline
    dataset = load_image_dataset(image_dir, streaming=True)
    first_item = next(iter(dataset))
    batch_sizes = [
        find_max_batch_size(pipeline, first_item["image"], 1, 100)
        for pipeline in pipelines
    ]
    logging.info(f"Using batch sizes: {batch_sizes}")

    # Start producer thread
    producer_thread = threading.Thread(target=producer, args=(image_dir, image_queue))

    # Start consumer threads
    consumer_threads = []
    for pipe, batch_size in zip(pipelines, batch_sizes):
        thread = threading.Thread(
            target=consumer, args=(pipe, image_queue, output_dir, batch_size)
        )
        consumer_threads.append(thread)

    # Start all threads
    producer_thread.start()
    for thread in consumer_threads:
        thread.start()

    # Wait for completion
    producer_thread.join()
    for thread in consumer_threads:
        thread.join()

    logging.info("Image processing completed")


if __name__ == "__main__":
    image_dir = "./input"
    output_dir = "./output"

    process_images_with_queue(image_dir=image_dir, output_dir=output_dir)
