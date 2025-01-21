# bulk-image-upscale

This is a python script to bulk upscale images. It runs on the gpu, and attempts to maximize the size of processed batches. It was tested with and uses swin2sr by default, but I don't see how another super-resolution model couldn't be used instead.

## Getting started

```bash
# clone the repo
git clone https://github.com/whiffymuffinz/bulk-image-upscale
cd bulk-image-upscale
# create a virtual environment
python -m venv venv
source venv/bin/activate
# install pytorch. this works for most users
pip install torch torchvision torchaudio
# install project specific requirements
pip install -r requirements.txt
# run the thing
python ./upscale_shared_queue.py
```

The current iteration of this code expects input images to be in a folder called `input`, and outputs upscaled images in the `output` directory.
This can be changed in the main function as well as most other parameters like the model that is used to upscale.
