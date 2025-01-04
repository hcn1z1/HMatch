import cv2
import os
import time
import matplotlib.pyplot as plt
import copy
import core.globals
from src.block_matching import build_pyramid, match_block, compute_residual,logarithmic_search
from tqdm import tqdm

FPS = 12
THRESHOLD = 70

def run_sequence_block(video_path, frame_idx, block_coords, second_frame_idx = None, block_size=16, search_window=16, pyramid_levels=3,output_dir = "results/",plot = False):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/sequence/", exist_ok=True)
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame1 = cap.read()
    frames = [frame1]
    start_coords = block_coords
    accuracy = []
    with tqdm(total=second_frame_idx,desc="Processing Frames",unit=" frame") as pbar:
        for i in range(second_frame_idx):
            ret, frame2 = cap.read()
            f1 = copy.deepcopy(frame1)
            f2 = frame2

            if not ret:
                raise ValueError(f"Unable to read frames at index {frame_idx}")

            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            

            # Extract the target block
            y, x = start_coords
            y2,x2 = block_coords
            target_block = frame1[y2:y2+block_size, x2:x2+block_size]

            pyramid = build_pyramid(frame2, pyramid_levels)
            if core.globals.algorithm in ['hierarchical', 'hier']: matched_coords, score = match_block(pyramid, target_block, (y, x), search_window, block_size)
            elif core.globals.algorithm in ["logarithmic","log"]: matched_coords, score = logarithmic_search(frame2, target_block, y, x, search_window, block_size)

            matched_block = frame2[matched_coords[0]:matched_coords[0] + block_size,
                                            matched_coords[1]:matched_coords[1] + block_size]
            score = abs(score)
            
            # Rotate result
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            cv2.rectangle(frame2, (matched_coords[1], matched_coords[0]),
                        (matched_coords[1]+block_size, matched_coords[0]+block_size), (0, 255, 0), 2)
            frames.append(frame2)
            if core.globals.score_algorithm == "NCC":
                accuracy.append(score)
                if score > 0.9:
                    frame1 = f2
                    block_coords = matched_coords 
                else : 
                    frame1 = f1
            else:
                residual = compute_residual(target_block, matched_block)

                score = residual.sum()/(block_size**2)
                accuracy.append(score)
                if THRESHOLD >= score:
                    frame1 = f2
                    block_coords = matched_coords 
                else:
                    frame1 = f1
                
            start_coords = matched_coords
                
            # Visualize
            
            pbar.update(1)
        
    # Output to video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = f"{output_dir}/sequence/{frame_idx}-{second_frame_idx}-{search_window}-{str(int(time.time()))[7::]}.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, FPS, (frames[0].shape[1],frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
        
    out.release()
    
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(list(range(second_frame_idx)), accuracy, marker='o', linestyle='-', color='b')
        plt.title('Accuracy of HMatch Across Iterations')
        plt.xlabel('Iteration Number')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.show()
            
    print(f"Video saved as {output_video_path} / {len(frames)} Frames")   
    
if __name__ == "__main__":
    run_sequence_block("data/video.mp4", frame_idx=10, block_coords=(32, 32))
