# Import Modules
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Process images from project_image_1.jpg to project_image_71.jpg
# Run another batch in 3O
for i in range(1, 29):
    # Construct the image filename
    filename = f'project_image_{i}.jpg'

    # Load the image
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Perform thresholding
    threshold_value: int = 90
    _, threshold_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Invert the thresholded image to create a binary image
    binary_image = cv2.bitwise_not(threshold_image)

    # Display the binary image
    plt.figure(figsize=(20, 15))
    plt.subplot(1, 1, 1)
    plt.imshow(binary_image, cmap='gray')
    plt.title(f'Threshold Image {i}')
    plt.show()

# try to fix this. im not sure if what I have done is correct##
# Perform connected component analysis
num_labels, labeled_image = cv2.connectedComponents(threshold_image)

# Feature Engineering
features = []
labels = []

for label in range(1, num_labels):
    # Extract region properties
    region_mask = (labeled_image == label).astype(np.uint8)  # Create a binary mask for the current region
    area = cv2.countNonZero(region_mask)  # Count non-zero pixels as area

    # Calculate circularity
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if contours else 0  # Use the contour's perimeter if available

    circularity = 0
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

    # You can add more feature calculations here

    # Classify based on rules (example rule: if circularity < 0.2, classify as crack)
    if circularity < 0.2:
        labels.append('crack')
    else:
        labels.append('no-crack')

# Visualize the connected components
plt.figure(figsize=(8, 6))
plt.imshow(labeled_image, cmap='jet')
plt.title(f'Connected Components: {num_labels - 1} Cracks Detected')

# Subtract 1 for the background
plt.colorbar()
plt.show()

# Print the labels for each region
for i, label in enumerate(labels):
    print(f'Region {i + 1}: {label}')
