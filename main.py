"""
Main function.
Spawns two threads for reading images, and feeding them to a pipeline, and another process for saving upscaled images to disk.
"""

import logging
import os
import queue
import threading

import torch
from tqdm import tqdm
from transformers import pipeline

from async_image_saver import async_image_saver
from utils import find_max_batch_size, load_image_dataset

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
        for image in tqdm(dataset_iterator, desc="items entered queue"):
            image_queue.put(image)
    except Exception as e:
        logging.error(f"Producer error: {e}")
    finally:
        # Signal the consumers to finish by adding None for each consumer
        for _ in range(torch.cuda.device_count()):
            image_queue.put(None)


def consumer(pipeline, image_queue, output_dir, batch_size, saver: async_image_saver):
    """
    Consumes image batches from the queue and processes them using the given pipeline.
    Implements OOM recovery by splitting failed batches into smaller chunks.

    Parameters:
    ----------
    pipeline : transformers.Pipeline
        Pipeline instance for processing images
    image_queue : queue.Queue
        Shared queue containing image batches
    output_dir : str
        Directory to save processed images
    batch_size : int
        Starting batch size for this consumer
    saver : async_image_saver
        Async image saver instance
    """
    device_index = (
        pipeline.model.device.index if hasattr(pipeline.model.device, "index") else -1
    )
    device_name = (
        torch.cuda.get_device_name(device_index) if device_index != -1 else "CPU"
    )

    while True:
        try:
            shutdown_flag = False
            batch = []

            # Accumulate a batch of images with timeout
            for _ in range(batch_size * 10):  # Adjust multiplier as needed
                image = image_queue.get(timeout=60)
                if image is None:
                    logging.info(f"Consumer on {device_name} received shutdown signal")
                    shutdown_flag = True
                    break
                batch.append(image)

            # Exit if shutdown signal received
            if shutdown_flag:
                break

            # Extract batch data
            batch_images = [item["image"] for item in batch]
            batch_paths = [item["path"] for item in batch]
            logging.info(
                f"Consumer on {device_name} processing batch of {len(batch_images)} "
                f"images (batch_size={batch_size})"
            )

            # Initialize variables for batch processing
            outputs = None
            processed_successfully = False

            try:
                # Attempt to process the full batch
                outputs = pipeline(batch_images, batch_size=batch_size)
                processed_successfully = True
                logging.info(f"Successfully processed batch on {device_name}")

            except torch.cuda.OutOfMemoryError as oom_error:
                logging.error(
                    f"OOM Error on {device_name} with batch size {batch_size} "
                    f"for {len(batch_images)} images: {oom_error}"
                )
                torch.cuda.empty_cache()

                # If batch contains single image and still OOMs, skip it
                if len(batch_images) <= 1:
                    logging.error(
                        f"OOM with single image on {device_name}. "
                        f"Skipping: {batch_paths[0]}"
                    )
                    continue

                # Attempt recovery by processing in smaller chunks
                chunk_size = max(
                    1, batch_size // 2
                )  # Start with half the original size
                recovery_attempts = 0
                max_recovery_attempts = 3  # Prevent infinite retries

                while recovery_attempts < max_recovery_attempts:
                    try:
                        recovered_outputs = []
                        logging.warning(
                            f"Attempting OOM recovery on {device_name} with "
                            f"chunk_size={chunk_size} (attempt {recovery_attempts + 1})"
                        )

                        # Process batch in chunks
                        for i in range(0, len(batch_images), chunk_size):
                            sub_batch = batch_images[i : i + chunk_size]
                            sub_outputs = pipeline(sub_batch, batch_size=chunk_size)
                            recovered_outputs.extend(sub_outputs)
                            torch.cuda.empty_cache()  # Clear cache between chunks

                        outputs = recovered_outputs
                        processed_successfully = True
                        logging.info(
                            f"Successfully recovered batch on {device_name} "
                            f"using chunk_size={chunk_size}"
                        )
                        break

                    except torch.cuda.OutOfMemoryError:
                        recovery_attempts += 1
                        chunk_size = max(1, chunk_size // 2)  # Halve chunk size again
                        torch.cuda.empty_cache()
                        if recovery_attempts >= max_recovery_attempts:
                            logging.error(
                                f"OOM recovery failed after {max_recovery_attempts} "
                                f"attempts on {device_name}. Skipping batch."
                            )
                            continue

            # Save processed images if successful
            if processed_successfully and outputs is not None:
                logging.info(
                    f"Saving {len(outputs)} processed images from {device_name}"
                )
                for path, output in zip(batch_paths, outputs):
                    output_path = os.path.join(output_dir, os.path.basename(path))
                    try:
                        saver.save(output, output_path)
                        logging.debug(f"Saved image to {output_path}")
                    except Exception as e:
                        logging.error(f"Failed to save image {path}: {str(e)}")

        except queue.Empty:
            logging.warning(f"Queue timeout on {device_name}")
            continue
        except Exception as e:
            logging.error(f"Consumer error on {device_name}: {e}")
            continue
        finally:
            image_queue.task_done()

    logging.info(f"Consumer on {device_name} shutting down")


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
        find_max_batch_size(pipeline, first_item["image"], 1, 5)
        for pipeline in tqdm(pipelines, desc="finding batch sizes")
    ]
    logging.info(f"Using batch sizes: {batch_sizes}")

    logging.info("Iitializing saver thread")
    saver = async_image_saver(output_dir)
    # Start producer thread
    logging.info("Initializing producer thread")
    producer_thread = threading.Thread(target=producer, args=(image_dir, image_queue))

    logging.info("Initializing consumer threads")
    # Start consumer threads
    consumer_threads = []
    for pipe, batch_size in zip(pipelines, batch_sizes):
        thread = threading.Thread(
            target=consumer, args=(pipe, image_queue, output_dir, batch_size, saver)
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

    saver.shutdown()

    logging.info("Image processing completed")


if __name__ == "__main__":
    IMAGE_DIR = "./input"
    OUTPUT_DIR = "./output"

    process_images_with_queue(
        image_dir=IMAGE_DIR, output_dir=OUTPUT_DIR, queue_size=1000
    )
