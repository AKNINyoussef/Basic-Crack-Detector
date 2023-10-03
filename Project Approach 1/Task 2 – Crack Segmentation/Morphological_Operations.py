import cv2
import numpy as np
import matplotlib.pyplot as plt

# Process images from project_image_1.jpg to project_image_27.jpg
for i in range(1, 28):
    # Construct the image filename
    filename = f'project_image_{i}.jpg'

    # Load the image
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Perform thresholding
    threshold_value = 90
    _, threshold_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Invert the thresholded image to create a binary image
    binary_image = cv2.bitwise_not(threshold_image)

    # Perform connected component analysis
    num_labels, labeled_image = cv2.connectedComponents(binary_image)

    # Feature Engineering
    labels = []  # Initialize a list to store region labels

    for label in range(1, num_labels):
        # Extract region properties
        region_mask = (labeled_image == label).astype(np.uint8)
        area = cv2.countNonZero(region_mask)

        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0

        circularity = 0
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)

        # Classify based on rules
        if circularity < 0.2:
            labels.append('crack')
        else:
            labels.append('no-crack')

    # Thin the binary image to a line-like representation of cracks
    thin_image = cv2.ximgproc.thinning(binary_image, thinningType=cv2.ximgproc.THINNING_GUOHALL)

    # Visualize the connected components and the thinned image
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(labeled_image, cmap='jet')
    plt.title(f'Connected Components: {num_labels - 1} Cracks Detected')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(thin_image, cmap='gray')
    plt.title('Thinned Image')

    plt.show()

    # Print the labels for each region
    for j, label in enumerate(labels):
        print(f'Region {j + 1} in {filename}: {label}')
