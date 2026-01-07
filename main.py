import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection

# 1. Load the color image using the function from image_utils.py
image_path = '/content/IMG_0158.webp'
original_image = load_image(image_path)

if original_image is not None:
    print(f"Image loaded successfully from {image_path}")
    print(f"Shape of original image: {original_image.shape}")

    # 2. Suppress noise in the loaded image using a median filter
    clean_image = median(original_image, ball(3))
    print("Noise suppression complete.")

    # 3. Run the noise-free image through the edge_detection function
    edge_magnitude_image = edge_detection(clean_image)
    print("Edge detection complete.")

    # 4. Convert the resulting edgeMAG array into a binary array
    # Display histogram to choose a threshold (optional, for manual selection)
    # plt.figure(figsize=(10, 6))
    # plt.hist(edge_magnitude_image.ravel(), bins=100, color='blue', alpha=0.7)
    # plt.title('Histogram of Edge Magnitude Image')
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()

    # Based on previous histogram analysis, a threshold of 50 is used
    threshold = 50
    edge_binary = edge_magnitude_image > threshold
    print(f"Binarization complete with threshold: {threshold}")

    # 5. Display the binary image and save it as a .png file
    # Convert boolean array to uint8 (0 or 255) for display and saving
    edge_binary_uint8 = (edge_binary * 255).astype(np.uint8)

    # Display the binary edge image
    plt.figure(figsize=(10, 8))
    plt.imshow(edge_binary_uint8, cmap='gray')
    plt.title('Binary Edge Image')
    plt.axis('off')
    plt.show()

    # Save the binary edge image as a PNG file
    output_filename = 'my_edges.png'
    edge_image = Image.fromarray(edge_binary_uint8)
    edge_image.save(output_filename)

    print(f"Binary edge image displayed and saved as '{output_filename}'")
else:
    print("Failed to load the image. Please check the path and ensure image_utils.py is correctly defined.")
