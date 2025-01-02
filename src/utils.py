import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(filepath, grayscale=True):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(filepath, flag)
    if image is None:
        raise FileNotFoundError(f"Image not found at {filepath}")
    return image

def extract_blocks(image, block_size):
    blocks = []
    coords = []
    h, w = image.shape
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            if block.shape == (block_size, block_size):  # Ensure full block size
                blocks.append(block)
                coords.append((y, x))
    return blocks, coords

def visualize_blocks(blocks, grid_shape, figsize=(10, 10)):
    fig, axs = plt.subplots(*grid_shape, figsize=figsize)
    for ax, block in zip(axs.ravel(), blocks):
        ax.imshow(block, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
