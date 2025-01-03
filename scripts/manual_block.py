import cv2
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from src.block_matching import build_pyramid, match_block

def run_manual_block(video_path, frame_idx, block_coords, second_frame_idx = None, block_size=16, search_window=16, pyramid_levels=3,output_dir = "results/",plot = False):
    os.makedirs(output_dir, exist_ok=True)
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    # Read the required frame and the next frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES,second_frame_idx) if second_frame_idx != None else None
    ret, frame2 = cap.read()
    f1 = copy.copy(frame1)
    f2 = copy.copy(frame2)

    if not ret:
        raise ValueError(f"Unable to read frames at index {frame_idx}")

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Extract the target block
    y, x = block_coords
    target_block = frame1[y:y+block_size, x:x+block_size]

    pyramid = build_pyramid(frame2, pyramid_levels)

    matched_coords, _ = match_block(pyramid, target_block, (y, x), search_window, block_size)

    # Output results
    print(f"Frame Index: {frame_idx}")
    print(f"Target block coordinates: {block_coords}")
    print(f"Matched block coordinates in next frame: {matched_coords}")

    # Visualize
    cv2.rectangle(f2, (matched_coords[1], matched_coords[0]),
                  (matched_coords[1]+block_size, matched_coords[0]+block_size), (0, 255, 0), 2)
    cv2.rectangle(f1,(block_coords[1], block_coords[0]),
                  (block_coords[1]+block_size, block_coords[0]+block_size), (0, 0, 255), 2)
    output_path = os.path.join(output_dir, f"annotated_{frame_idx}.png")
    cv2.imwrite(output_path, f2)
    if plot:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title('Image 1 - Reference')
        plt.imshow(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB))

        plt.subplot(1,2,2)
        plt.title('Image 2 - Test with Matched Blocks')
        plt.imshow(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
        plt.show()


if __name__ == "__main__":
    run_manual_block("data/video.mp4", frame_idx=10, block_coords=(32, 32))
