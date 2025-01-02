import cv2
import numpy as np
from tqdm import tqdm

THRESHOLD = 30 # Lower is better for Residual score

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
    Match a block at each pyramid level.
    """
    best_coords = start_coords
    best_score = float('inf')  # Lower is better for error metric
    h, w = target_block.shape

    for level, image in enumerate(reversed(pyramid)):
        scale = 2 ** (len(pyramid) - level - 1)
        scaled_coords = (best_coords[0] // scale, best_coords[1] // scale)
        scaled_window = search_window // scale

        # Define search area
        y_start = max(scaled_coords[0] - scaled_window, 0)
        y_end = min(scaled_coords[0] + scaled_window + block_size, image.shape[0])
        x_start = max(scaled_coords[1] - scaled_window, 0)
        x_end = min(scaled_coords[1] + scaled_window + block_size, image.shape[1])

        # Search within the area
        for y in range(y_start, y_end - h + 1):
            for x in range(x_start, x_end - w + 1):
                candidate_block = image[y:y + h, x:x + w]
                score = np.sum(np.abs(target_block - candidate_block))  # Sum of Absolute Differences (SAD)
                if score < best_score:
                    best_score = score
                    best_coords = (y * scale, x * scale)

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
    with tqdm(total=len(blocks), desc="Processing Blocks", unit=" block") as pbar:
        for block, coord in zip(blocks, coords):
            matched_coord, _ = match_block(pyramid, block, coord, search_window, block_size)
            matched_block = reference_frame[matched_coord[0]:matched_coord[0] + block_size,
                                            matched_coord[1]:matched_coord[1] + block_size]
            residual = compute_residual(block, matched_block)

            score = residual.sum()/(block_size**2)
            if THRESHOLD >= score:
                matched_blocks.append(matched_coord)
                residuals.append(residual)
            pbar.update(1)

    return matched_blocks, residuals

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
