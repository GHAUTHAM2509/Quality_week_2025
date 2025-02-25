#!/bin/bash

# Directory containing the videos
VIDEO_DIR="Video_train/AI"
# Directory to store the images
IMAGE_DIR="Video_train/AI_img"

# Create the image directory if it doesn't exist
mkdir -p "$IMAGE_DIR"

# Loop through the first 10 video files in the directory
for video in "$VIDEO_DIR"/*.{mp4,mov,avi,mkv}; do
  # Extract the base name of the video file
  base_name=$(basename "$video")
  # Remove the file extension
  base_name="${base_name%.*}"

  # Use ffmpeg to split the video into images
  ffmpeg -i "$video" -vf "fps=5" "$IMAGE_DIR/${base_name}_%04d.jpg"
done