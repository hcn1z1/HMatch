import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import logging
from src.block_matching import build_pyramid, match_block, compute_residual, process_frame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_image_with_circle(image_size, circle_radius, circle_color, background_color, shift=(0,0)):
    """
    Create an image with a circle at a random position, optionally shifted.
    """
    img = np.full((image_size[0], image_size[1], 3), background_color, dtype=np.uint8)
    
    # Ensure the circle fits within the image boundaries after shift
    margin = circle_radius + max(abs(shift[0]), abs(shift[1]))
    center_x = random.randint(margin, image_size[1] - margin)
    center_y = random.randint(margin, image_size[0] - margin)
    
    # Apply shift
    shifted_center = (center_y + shift[0], center_x + shift[1])
    
    # Draw the circle
    cv2.circle(img, (shifted_center[1], shifted_center[0]), circle_radius, circle_color, -1)
    
    return img, shifted_center

def convert_to_grayscale(image):
    """
    Convert a BGR image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def main_demo():
    IMAGE_SIZE = (256, 256)
    CIRCLE_RADIUS = 20
    CIRCLE_COLOR = (0, 255, 0)
    BACKGROUND_COLOR = (0, 0, 0)
    SHIFT = (0, 0)

    image1, center1 = create_image_with_circle(
        IMAGE_SIZE, CIRCLE_RADIUS, CIRCLE_COLOR, BACKGROUND_COLOR, shift=(0, 0)
    )
    image2, center2 = create_image_with_circle(
        IMAGE_SIZE, CIRCLE_RADIUS, CIRCLE_COLOR, BACKGROUND_COLOR, shift=SHIFT
    )

    logging.info(f"Circle Center in Image 1: {center1}")
    logging.info(f"Circle Center in Image 2: {center2}")

    gray1 = convert_to_grayscale(image1)
    gray2 = convert_to_grayscale(image2)

    block_size = 50  
    search_window = -1
    pyramid_levels = 3
    
    pyramid = build_pyramid(gray2, pyramid_levels)
    block_coords = [int(center1[1] - CIRCLE_RADIUS),int(center1[0]-CIRCLE_RADIUS)]
    target_block = gray1[
        block_coords[1]:block_coords[1] + block_size,
        block_coords[0]:block_coords[0] + block_size
    ]
    matched_coords, _ = match_block(pyramid, target_block, (int(center1[0]-CIRCLE_RADIUS), int(center1[1]-CIRCLE_RADIUS)), search_window, block_size)
    logging.info(f"Matched coords : {matched_coords}")
    
    matched_blocks, residuals,original_coords = process_frame(
        gray1, gray2, block_size, search_window, pyramid_levels
    )

    logging.info(f"Number of matched blocks: {len(matched_blocks)}")
    logging.info(f"Number of residuals: {len(residuals)}")
    image2_matched = image2.copy()
    cv2.rectangle(image2_matched, (matched_coords[1], matched_coords[0]),
                  (matched_coords[1]+block_size, matched_coords[0]+block_size), (0, 255, 0), 2)
    
    cv2.rectangle(image1,(block_coords[0], block_coords[1]),
                  (block_coords[0]+block_size, block_coords[1]+block_size), (0, 0, 255), 2)
    

    for block_coords,original in zip(matched_blocks,original_coords):
        if block_coords != (0,0):
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            y, x = block_coords
            top_left = (x , y)
            bottom_right = (x  + block_size, y + block_size)
            cv2.rectangle(image2_matched, top_left, bottom_right, color, 2)  # Blue rectangles
            y, x = original
            top_left = (x , y)
            bottom_right = (x  + block_size, y + block_size)
            cv2.rectangle(image1, top_left, bottom_right, color, 2) 

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('Image 1 - Reference')
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.scatter(center1[1], center1[0], color='red', marker='x')  # Mark the center

    plt.subplot(1,2,2)
    plt.title('Image 2 - Test with Matched Blocks')
    plt.imshow(cv2.cvtColor(image2_matched, cv2.COLOR_BGR2RGB))
    plt.scatter(center2[1], center2[0], color='red', marker='x')  # Mark the center
    plt.show()

if __name__ == "__main__":
    main_demo()
