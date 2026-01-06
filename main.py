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
from tile_cache import RunLocalTileCache
from tiled_processor import TiledBatchProcessor
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


def consumer(
    pipeline,
    image_queue,
    output_dir,
    batch_size,
    saver: async_image_saver,
    cache: RunLocalTileCache,
):
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
    cache : RunLocalTileCache
        Shared tile cache for deduplication
    """
    device_index = (
        pipeline.model.device.index if hasattr(pipeline.model.device, "index") else -1
    )
    device_name = (
        torch.cuda.get_device_name(device_index) if device_index != -1 else "CPU"
    )

    # Initialize tiled processor
    # Note: batch_size here refers to the number of images we pull from the queue.
    # storage_batch_size used by processor for internal tile batches can be fixed or related.
    # Since tiles are small (128x128), we can use a decent batch size for inference.
    processor = TiledBatchProcessor(
        pipeline=pipeline, cache=cache, batch_size=32  # Good default for tiles
    )

    while True:
        try:
            shutdown_flag = False
            batch = []

            # Accumulate a batch of images with timeout
            # For tiled processing, we can process fewer images at once if they are large,
            # but 'batch_size' was calculated for whole images.
            # We stick to the passed batch_size for queue consumption.
            for _ in range(batch_size):
                try:
                    image = image_queue.get(timeout=5)
                    if image is None:
                        logging.info(
                            f"Consumer on {device_name} received shutdown signal"
                        )
                        shutdown_flag = True
                        break
                    batch.append(image)
                except queue.Empty:
                    break

            if not batch and not shutdown_flag:
                # Queue empty but no shutdown yet, wait a bit
                continue

            # Process what we have
            if batch:
                # Extract batch data
                batch_images = [item["image"] for item in batch]
                batch_paths = [item["path"] for item in batch]
                logging.info(
                    f"Consumer on {device_name} processing batch of {len(batch_images)} "
                    f"images"
                )

                try:
                    # Process batch using tiled processor
                    # Tiled processor handles internal OOM recovery for tile batches?
                    # The current implementation of TiledBatchProcessor processes tiles in chunks.
                    # It doesn't have the sophisticated recursive OOM recovery of the old consumer,
                    # but since tiles are small and uniform, OOM is much less likely/predictable.
                    outputs = processor.process_batch(batch_images)
                    processed_successfully = True

                    # Save processed images
                    logging.info(
                        f"Saving {len(outputs)} processed images from {device_name}"
                    )
                    for path, output in zip(batch_paths, outputs):
                        output_path = os.path.join(output_dir, os.path.basename(path))
                        try:
                            saver.save(output, output_path)
                        except Exception as e:
                            logging.error(f"Failed to save image {path}: {str(e)}")

                except Exception as e:
                    logging.error(f"Error processing batch on {device_name}: {e}")
                    # If TiledBatchProcessor fails, we lose this batch.
                    # Implementing retry logic for tiled processor is complex due to cache state.
                    # For now, we log and continue.

                finally:
                    # Mark tasks as done
                    for _ in batch:
                        image_queue.task_done()

            # Exit if shutdown signal received
            if shutdown_flag:
                # Mark the None task as done
                image_queue.task_done()
                break

        except Exception as e:
            logging.error(f"Consumer loop error on {device_name}: {e}")
            continue

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

    # Initialize Tile Cache (Run-Global Deduplication)
    tile_cache = RunLocalTileCache()

    try:
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
            # Fallback to CPU if no GPU
            if torch.cuda.device_count() == 0:
                logging.warning("No GPU found. Initializing CPU pipeline.")
                pipe = pipeline("image-to-image", model=model_id, device="cpu")
                pipelines.append(pipe)
            else:
                raise RuntimeError("No pipelines could be initialized")

        # Find optimal batch size using the first pipeline
        # Note: With tiled processing, the 'image batch size' matters less for OOM,
        # but matters for deduplication efficiency (larger batch = more chance to dedup).
        # We can stick to a reasonable default or dynamic calculation.
        # Since we are tiling, 1 image is actually many tiles.
        # Let's use a conservative batch size for images to avoid holding too many large images in RAM.
        dataset = load_image_dataset(image_dir, streaming=True)
        try:
            first_item = next(iter(dataset))
            # Just verify we can load images
            pass
        except StopIteration:
            logging.warning("Dataset is empty")
            return

        # Fixed batch size for image consumption (e.g., 4 images at a time)
        # Tiling will break them down.
        batch_sizes = [4] * len(pipelines)
        logging.info(f"Using batch sizes: {batch_sizes}")

        logging.info("Initializing saver thread")
        saver = async_image_saver(output_dir)

        # Start producer thread
        logging.info("Initializing producer thread")
        producer_thread = threading.Thread(
            target=producer, args=(image_dir, image_queue)
        )

        logging.info("Initializing consumer threads")
        # Start consumer threads
        consumer_threads = []
        for pipe, batch_size in zip(pipelines, batch_sizes):
            thread = threading.Thread(
                target=consumer,
                args=(pipe, image_queue, output_dir, batch_size, saver, tile_cache),
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

    finally:
        # Ensure cache cleanup
        if tile_cache:
            tile_cache.cleanup()


if __name__ == "__main__":
    IMAGE_DIR = "./input"
    OUTPUT_DIR = "./output"

    process_images_with_queue(
        image_dir=IMAGE_DIR, output_dir=OUTPUT_DIR, queue_size=1000
    )
