import cv2
import os
import random
import matplotlib.pyplot as plt
import core.globals
from src.block_matching import process_frame
from src.utils import load_image

def run_frame_blocks(video_path, frame_idx, second_frame_idx = None,block_size=16, search_window=16, pyramid_levels=3,output_dir = "results/", plot = False):
    # Load the video
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/annotations/', exist_ok=True)
    os.makedirs(f'{output_dir}/frame/', exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    # Read the required frame and the next frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame1 = cap.read()
    if second_frame_idx != None: cap.set(cv2.CAP_PROP_POS_FRAMES,second_frame_idx) 
    ret, frame2 = cap.read()

    if not ret:
        raise ValueError(f"Unable to read frames at index {frame_idx}")

   
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    matched_blocks, residuals , original_blocks = process_frame(frame1, frame2, block_size, search_window, pyramid_levels)

    print(f"Processed {len(matched_blocks)} blocks in frame {frame_idx}")
    for idx, (coord, residual) in enumerate(zip(matched_blocks, residuals)):
        print(f"Block {idx}: Matched at {coord}, Residual sum: {residual.sum()}")
        
    for (row, col), original  in zip(matched_blocks, original_blocks):
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        top_left     = (col, row)
        bottom_right = (col + block_size, row + block_size)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        cv2.rectangle(
            frame2,
            top_left,
            bottom_right,
            color,
            2 
        )
        y, x = original
        top_left = (x , y)
        bottom_right = (x  + block_size, y + block_size)
        cv2.rectangle(frame1, top_left, bottom_right, color, 2) 
        
    output_path = os.path.join(output_dir,'annotations/', f"annotated_{second_frame_idx if second_frame_idx!=None else frame_idx+1}.png")
    cv2.imwrite(output_path, frame2)
    output_path = os.path.join(output_dir,'annotations/', f"original_{second_frame_idx if second_frame_idx!=None else frame_idx+1}.png")
    cv2.imwrite(output_path, frame1)
    print(f"Saved annotated frame to {output_path}")
    if plot:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title('Image 1 - Reference')
        plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

        plt.subplot(1,2,2)
        plt.title('Image 2 - Test with Matched Blocks')
        plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        plt.show()
    
if __name__ == "__main__":
    run_frame_blocks("data/video.mp4", frame_idx=10)
