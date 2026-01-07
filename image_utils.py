import numpy as np
from PIL import Image

def load_image(file_path):
    """
    Loads a color image from a file path and converts it to a NumPy array.

    Args:
        file_path (str): The path to the image file.

    Returns:
        np.array: The image as a NumPy array, or None if an error occurs.
    """
    try:
        img = Image.open(file_path)
        return np.array(img)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return None

# Test the function
image_path = '/content/IMG_0158.webp'  # Using the selected image path
loaded_img_array = load_image(image_path)

if loaded_img_array is not None:
    print(f"Image loaded successfully from {image_path}")
    print(f"Shape of the loaded image: {loaded_img_array.shape}")
    print(f"Data type of the loaded image: {loaded_img_array.dtype}")
    print("First 5x5 pixels of the image (top-left corner, first channel):")
    print(loaded_img_array[:5, :5, 0])
else:
    print("Failed to load image.")

import numpy as np
from scipy.signal import convolve2d

def edge_detection(image_array):
    """
    Performs edge detection on a color image array.

    Args:
        image_array (np.array): A 3-channel color image as a NumPy array.

    Returns:
        np.array: The edge magnitude array (edgeMAG).
    """
    # 1. Convert to grayscale
    # Convert image_array to float to avoid overflow during averaging
    grayscale_image = np.mean(image_array.astype(float), axis=2)

    # 2. Create kernelY for vertical changes
    kernelY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    # 3. Create kernelX for horizontal changes
    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # 4. Apply filters using convolve2d with zero padding ('same' mode)
    # The output will have the same size as the input
    edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)

    # 5. Compute edgeMAG
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG

# Test the function
if 'loaded_img_array' in locals() and loaded_img_array is not None:
    print(f"Original image shape for edge detection: {loaded_img_array.shape}")
    edge_magnitude_image = edge_detection(loaded_img_array)

    print(f"Edge magnitude image shape: {edge_magnitude_image.shape}")
    print(f"Edge magnitude image data type: {edge_magnitude_image.dtype}")
    print("First 5x5 values of the edge magnitude image:")
    print(edge_magnitude_image[:5, :5])
else:
    print("Please run the load_image function first to get 'loaded_img_array'.")



