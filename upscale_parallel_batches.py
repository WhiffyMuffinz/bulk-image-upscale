"""
github.com/whiffymuffinz
gpl license I guess.
batches are handled incorrectly, but whatever
"""


import random
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
import torch
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [
            os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            return {"image": image, "path": img_path}
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None


def setup_pipelines(model_id="caidas/swin2SR-lightweight-x2-64"):
    pipelines = []
    devices = [
        f"cuda:{i}" for i in range(torch.cuda.device_count())
    ] if torch.cuda.is_available() else ["cpu"]

    for device in devices:
        try:
            pipelines.append(
                pipeline(
                    "image-to-image",
                    model=model_id,
                    device=device,
                    torch_dtype=torch.float16 if "cuda" in device else None
                )
            )
        except Exception as e:
            print(f"Failed to initialize pipeline on {device}: {e}")

    return pipelines


def find_max_batch_size(pipe, test_image, min_size=0, max_size=100, method='binary'):
    """
    Find the maximum batch size that can fit in GPU memory using either linear or binary search.

    Args:
        pipe: The pipeline or model to test
        test_image: A single test image to use for batch size testing
        min_size: Minimum batch size to try (default: 0)
        max_size: Maximum batch size to try (default: 100)
        method: Search method - 'binary' or 'linear' (default: 'binary')

    Returns:
        int: Maximum batch size that fits in GPU memory
    """
    def test_batch_size(size):
        try:
            # Create a new batch of images
            # Each image should be loaded fresh to actually consume memory
            batch_images = []
            for _ in range(size):
                # Here we load a fresh copy of the image instead of just referencing it
                # Assuming test_image is a path or a function that loads an image
                if isinstance(test_image, str):
                    # If it's a path, load the image fresh each time
                    img = Image.open(test_image)
                else:
                    # If it's a function or other loader, call it to get a fresh image
                    img = test_image() if callable(test_image) else test_image.copy()
                batch_images.append(img)
            print("size: ", size)
            # Test the batch
            pipe(batch_images, batch_size=size)
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            raise

    if method == 'linear':
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
                # This size works, try a larger one
                best_size = mid
                left = mid + 1
            else:
                # Too big, try a smaller size
                right = mid - 1

        return best_size

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {"image": [], "path": []}
    return {
        "image": [item["image"] for item in batch],
        "path": [item["path"] for item in batch]
    }


def process_images_on_device(device_index, subset, pipeline, batch_size, output_dir):
    dataloader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    os.makedirs(output_dir, exist_ok=True)

    for batch in tqdm(dataloader, desc=f"Device {device_index}"):
        if not batch["image"]:
            continue

        outputs = pipeline(batch["image"], batch_size=len(batch["image"]))

        for path, output in zip(batch["path"], outputs):
            output_path = os.path.join(output_dir, os.path.basename(path))
            if isinstance(output, Image.Image):
                output.save(output_path)
            else:
                Image.fromarray(output).save(output_path)


def process_images_with_parallel_devices(image_dir, pipelines, batch_sizes, output_dir="./output"):
    dataset = ImageDataset(image_dir)
    num_devices = len(pipelines)
    chunk_size = len(dataset) // num_devices
    subsets = [
        Subset(dataset, range(i * chunk_size, (i + 1) * chunk_size))
        for i in range(num_devices)
    ]

    # Handle leftover data by assigning it to the last subset
    if len(dataset) % num_devices > 0:
        subsets[-1] = Subset(
            dataset,
            range((num_devices - 1) * chunk_size, len(dataset))
        )

    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        futures = [
            executor.submit(
                process_images_on_device,
                device_index=i,
                subset=subsets[i],
                pipeline=pipelines[i],
                batch_size=batch_sizes[i],
                output_dir=os.path.join(output_dir, f"device_{i}")
            )
            for i in range(num_devices)
        ]

        for future in tqdm(futures, desc="Overall Progress"):
            future.result()


if __name__ == "__main__":
    model_id = "caidas/swin2SR-lightweight-x2-64"
    image_dir = "./input"
    output_dir = "./output"

    # Setup pipelines for all available devices
    pipelines = setup_pipelines(model_id=model_id)

    # Load a random image from the dataset for testing
    dataset = ImageDataset(image_dir)
    random_index = random.randint(0, len(dataset) - 1)
    random_test_image = dataset[random_index]["image"]

    # Test batch size for each pipeline
    batch_sizes = [find_max_batch_size(pipe, random_test_image,1, 11 ) for pipe in pipelines]

    # Process images in parallel on all devices
    process_images_with_parallel_devices(
        image_dir=image_dir,
        pipelines=pipelines,
        batch_sizes=batch_sizes,
        output_dir=output_dir
    )
