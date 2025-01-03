import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import core.globals
from sklearn.metrics import mutual_info_score


THRESHOLD = 70 # Lower is better for Residual score
DEBUG = False # Debug mode will be useful to track heatmap

import numpy as np

def compute_ssd(block1, block2):
    """
    Computes the Sum of Squared Differences (SSD) between two blocks.
    """
    return np.sum((block1 - block2)**2)

def compute_ncc(block1, block2):
    '''
    Testing NCC if it is better than SAD
    '''
    mean1 = np.mean(block1)
    mean2 = np.mean(block2)
    numerator = np.sum((block1 - mean1) * (block2 - mean2))
    denominator = np.sqrt(np.sum((block1 - mean1)**2) * np.sum((block2 - mean2)**2))
    return numerator / denominator if denominator != 0 else -1

def compute_mutual_information(block1, block2):
    """
    Computes the Mutual Information (MI) between two image blocks.
    This measures the amount of information one block shares with the other.
    """
    # Flatten the blocks
    block1 = block1.flatten()
    block2 = block2.flatten()

    block1 = (block1 - np.min(block1)) / (np.max(block1) - np.min(block1) + 1e-10)
    block2 = (block2 - np.min(block2)) / (np.max(block2) - np.min(block2) + 1e-10)
    block1 = (block1 * 255).astype(int)
    block2 = (block2 * 255).astype(int)
    mi = mutual_info_score(block1, block2)
    h1 = mutual_info_score(block1, block1)
    h2 = mutual_info_score(block2, block2) 
    nmi = mi / np.sqrt(h1 * h2) if h1 > 0 and h2 > 0 else 0
    
    return mi

def build_pyramid(image, levels):
    """
    Build an image pyramid for hierarchical processing.
    """
    pyramid = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def match_block(pyramid, target_block, start_coords, search_window, block_size=16):
    """
    Match a block at each pyramid level. Searches the entire image if search_window is set to -1.

    Args:
        pyramid: List of images at different pyramid levels.
        target_block: The block to match (from the target frame).
        start_coords: Coordinates of the initial block.
        search_window: Size of the search window. Use -1 for full image search.
        block_size: Size of the block (e.g., 16x16).

    Returns:
        best_coords: Coordinates of the best-matched block.
        best_score: Best matching score (lower is better for NCC or SAD).
    """
    best_coords = start_coords
    best_score = float('inf')  # Lower is better for NCC or SAD
    h, w = target_block.shape

    for level, image in enumerate(reversed(pyramid)):
        scale = 2 ** (len(pyramid) - level - 1)
        scaled_coords = (best_coords[0] // scale, best_coords[1] // scale)
        scaled_window = search_window // scale if search_window != -1 else image.shape[0]

        # Define search area for the current pyramid level
        y_start = 0 if search_window == -1 else max(scaled_coords[0] - scaled_window, 0)
        y_end = image.shape[0] if search_window == -1 else min(scaled_coords[0] + scaled_window + block_size, image.shape[0])
        x_start = 0 if search_window == -1 else max(scaled_coords[1] - scaled_window, 0)
        x_end = image.shape[1] if search_window == -1 else min(scaled_coords[1] + scaled_window + block_size, image.shape[1])

        # Search within the area
        if DEBUG: heatmap = np.zeros((image.shape[0], image.shape[1]))
        for y in range(y_start, y_end - h + 1):
            for x in range(x_start, x_end - w + 1):
                candidate_block = image[y:y + h, x:x + w]
                if core.globals.score_algorithm == "NCC":score = -compute_ncc(target_block, candidate_block)  # NCC: Higher is better, so negate
                elif core.globals.score_algorithm == "MI": score = -compute_mutual_information(target_block,candidate_block) # Higher is better so negate too
                elif core.globals.score_algorithm == "SSD": score = compute_ssd(target_block,candidate_block)
                else: score = np.sum(np.abs(target_block - candidate_block))  # Use SAD if preferred
                if DEBUG: heatmap[y, x] = score
                if score < best_score:
                    best_score = score
                    best_coords = (y * scale, x * scale)
        
        if DEBUG: print(f"Level {level}: Best Coords = {best_coords}, Best Score = {best_score}")

        # Visualize the heatmap for debugging
        if DEBUG:
            plt.title(f"Heatmap at Pyramid Level {level}")
            plt.imshow(heatmap, cmap='hot')
            plt.colorbar(label="Matching Score")
            plt.show()

    return best_coords, best_score


def compute_residual(block1, block2):
    """
    Compute the difference between two blocks.
    """
    return block1 - block2

def process_frame(frame, reference_frame, block_size=16, search_window=16, pyramid_levels=3):
    """
    Process all blocks in a frame and match them to the reference frame.
    """
    blocks, coords = extract_blocks(frame, block_size)
    pyramid = build_pyramid(reference_frame, pyramid_levels)
    matched_blocks = []
    residuals = []
    origin_block = []
    with tqdm(total=len(blocks), desc="Processing Blocks", unit=" block") as pbar:
        for block, coord in zip(blocks, coords):
            matched_coord, _ = match_block(pyramid, block, coord, search_window, block_size)
            matched_block = reference_frame[matched_coord[0]:matched_coord[0] + block_size,
                                            matched_coord[1]:matched_coord[1] + block_size]
            residual = compute_residual(block, matched_block)

            score = residual.sum()/(block_size**2)
            if THRESHOLD >= score:
                matched_blocks.append(matched_coord)
                origin_block.append(coord)
                residuals.append(residual)
            pbar.update(1)

    return matched_blocks, residuals,origin_block

def extract_blocks(image, block_size):
    """
    Extract non-overlapping blocks from the image.
    """
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
