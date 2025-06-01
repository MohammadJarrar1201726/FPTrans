from PIL import Image
import numpy as np

# Read paths from output.txt
with open('../output.txt', 'r') as f:
    image_paths = [line.strip() for line in f if line.strip()]

# Track invalid images (those that contain only 0s)
invalid_images = []

# Loop through each image path
for path in image_paths:
    try:
        img = Image.open(path).convert('L')  # convert to grayscale
        img_np = np.array(img)

        if not (img_np == 255).any():  # Check if there is no pixel with value 1
            invalid_images.append(path)

    except Exception as e:
        print(f"Error processing {path}: {e}")
        invalid_images.append(path)

# Output result
if invalid_images:
    print("The following images do NOT contain any pixel with value 1:")
    print(len(invalid_images))
    for img in invalid_images:
        print(img)
else:
    print("✅ All images contain at least one pixel with value 1.")
