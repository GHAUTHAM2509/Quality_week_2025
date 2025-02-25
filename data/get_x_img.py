import os
import random
import shutil

# Directory containing the images
SOURCE_DIR = 'Video_train/real_img'
# Directory to store the selected images
DEST_DIR = 'vt25/real'

# Number of images to select
NUM_IMAGES = 25000

# Create the destination directory if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

# Get a list of all image files in the source directory
all_images = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]

# Randomly select the specified number of images
selected_images = random.sample(all_images, NUM_IMAGES)

# Copy the selected images to the destination directory
for image in selected_images:
    shutil.copy(os.path.join(SOURCE_DIR, image), os.path.join(DEST_DIR, image))

print(f'Successfully copied {NUM_IMAGES} images to {DEST_DIR}')