#!/bin/bash

# Get the metadata from the input file
# Usage: ./script.sh <input_reference_file>
set -e

if [ $# -lt 1 ]; then
  echo "Error: Please provide an input reference file"
  echo "Usage: $0 <input_reference_file>"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="output_video.mkv"
FRAME_PATTERN="./output/frame_%06d.png"

# Get the frame rate from the input file
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$INPUT_FILE")

# Create a temporary video with just the HuffYUV video stream
ffmpeg -framerate $FPS \
  -i "$FRAME_PATTERN" \
  -c:v ffv1 \
  temp_video.mkv

# Combine the new video stream with all streams from the original file
ffmpeg -i "$INPUT_FILE" \
  -i temp_video.mkv \
  -map 1:v:0 \
  -map 0:a? \
  -map 0:s? \
  -map 0:t? \
  -map 0:d? \
  -c copy \
  "$OUTPUT_FILE"

# Clean up temporary file
rm temp_video.mkv

# Verify the output
echo "Video created: $OUTPUT_FILE"
ffprobe -v error -show_format -show_streams "$OUTPUT_FILE"
