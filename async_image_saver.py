import logging
import multiprocessing as mp
from queue import Empty
import os
from pathlib import Path
from PIL import Image


class async_image_saver:
    def __init__(self, output_dir: str, max_queue_size: int = 100):
        """
        Initialize async image saver

        Args:
            output_dir: Directory to save processed images
            max_queue_size: Maximum number of items in save queue
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create queue for communication between processes
        self.queue = mp.Queue(maxsize=max_queue_size)

        # Start save process
        self.save_process = mp.Process(
            target=self._save_worker, args=(self.queue, self.output_dir)
        )
        self.save_process.start()
        logging.info(f"Started async image saver process for {output_dir}")

    @staticmethod
    def _save_worker(queue: mp.Queue, output_dir: Path):
        """Worker process that handles saving images"""
        while True:
            try:
                # Get data from queue with timeout
                data = queue.get(timeout=1)

                if data is None:  # Shutdown signal
                    break

                image, output_path = data
                try:
                    image.save(output_path)
                    logging.debug(f"Saved processed image to {output_path}")
                except Exception as e:
                    logging.error(f"Failed to save image {output_path}: {str(e)}")

            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error in save worker: {e}")

    def save(self, image: Image.Image, path: str):
        """Queue an image for saving. Non-blocking operation."""
        output_path = self.output_dir / os.path.basename(path)
        self.queue.put((image, output_path))

    def shutdown(self):
        """Gracefully shutdown the save process"""
        self.queue.put(None)  # Send shutdown signal
        self.save_process.join()
        logging.info("Shut down async image saver process")
