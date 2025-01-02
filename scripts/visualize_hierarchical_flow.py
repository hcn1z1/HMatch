import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from src.block_matching import build_pyramid, match_block

def visualize_hierarchical_flow(
    video_path,
    frame_idx,
    block_coords,
    block_size=16,
    search_window=16,
    pyramid_levels=3,
    second_frame_idx = None
):
    # --- 1) Read the frames ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES,second_frame_idx) if second_frame_idx != None else None
    ret, frame2 = cap.read()
    if not ret:
        raise ValueError(f"Unable to read frames at index {frame_idx}")
    cap.release()

    # Convert to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # --- 2) Build pyramids ---
    # pyramid[0] is the full-resolution image, pyramid[-1] is the coarsest
    pyramid1 = build_pyramid(frame1_gray, pyramid_levels)
    pyramid2 = build_pyramid(frame2_gray, pyramid_levels)

    # --- 3) We'll match from top (coarsest) level down to the bottom (finest) ---
    # pyramid_levels = 3 => top level index = 2, bottom level index = 0

    top_level = pyramid_levels - 1  # e.g. level 2
    # "scale_factor" from the bottom to top is 2^(top_level)
    # So to get top-level coords, we scale the full-res coords down by 2^(top_level).
    scale_top = 2 ** top_level
    top_coords = (block_coords[0] // scale_top, block_coords[1] // scale_top)
    top_block_size = block_size // scale_top
    top_search_window = search_window // scale_top

    # Extract block in pyramid1 at top_level
    reference_block = pyramid1[top_level][
        top_coords[0] : top_coords[0] + top_block_size,
        top_coords[1] : top_coords[1] + top_block_size
    ]

    # --- 4) Initialize a directed graph to visualize the flow ---
    G = nx.DiGraph()
    pos = {}  # node -> (x,y)
    node_labels = {}

    # --- 5) Match at the top level ---
    matched_top, _ = match_block(
        [pyramid2[top_level]],
        reference_block,
        top_coords,
        top_search_window,
        top_block_size
    )
    # Create a node for the top-level match
    node_id = f"Level {top_level} -> {matched_top}"
    G.add_node(node_id)
    pos[node_id] = (top_level, -matched_top[0])  # For layout: x=level, y=-row
    node_labels[node_id] = f"({matched_top[0]}, {matched_top[1]})"

    # We will keep track of the previously matched coords
    prev_coords = matched_top
    prev_node_id = node_id

    # Also keep track of the block we will “refine” at each level
    target_block = pyramid2[top_level][
        matched_top[0] : matched_top[0] + top_block_size,
        matched_top[1] : matched_top[1] + top_block_size
    ]

    # --- 6) Descend from top_level-1 down to 0 ---
    for level in range(top_level - 1, -1, -1):
        # Going from level+1 => level => the scale factor changes by 2
        # So we "scale up" our matched coords from the coarser level by *2*
        refined_coords = (prev_coords[0] * 2, prev_coords[1] * 2)
        block_size_level = block_size // (2 ** level)
        search_window_level = search_window // (2 ** level)

        # Re-extract the reference block from pyramid1 at this finer level
        reference_block = pyramid1[level][
            refined_coords[0] : refined_coords[0] + block_size_level,
            refined_coords[1] : refined_coords[1] + block_size_level
        ]

        # Match it in pyramid2 at the same level
        matched_level, _ = match_block(
            [pyramid2[level]],
            reference_block,
            refined_coords,
            search_window_level,
            block_size_level
        )

        # Create a node for the newly matched coords
        node_id = f"Level {level} -> {matched_level}"
        G.add_node(node_id)
        pos[node_id] = (level, -matched_level[0])
        node_labels[node_id] = f"({matched_level[0]}, {matched_level[1]})"

        # Connect from the previous node to this new node
        G.add_edge(prev_node_id, node_id)

        # Update for next iteration
        prev_coords = matched_level
        prev_node_id = node_id
        target_block = pyramid2[level][
            matched_level[0] : matched_level[0] + block_size_level,
            matched_level[1] : matched_level[1] + block_size_level
        ]

    # --- 7) Highlight the final (lowest-level) node in red ---
    G.nodes[prev_node_id]['color'] = 'red'

    # Pick the color for each node
    node_colors = [
        'red' if ('color' in G.nodes[n] and G.nodes[n]['color'] == 'red')
        else 'blue'
        for n in G.nodes
    ]

    # --- 8) Draw the graph ---
    plt.figure(figsize=(7, 5))
    nx.draw(
        G,
        pos,
        labels=node_labels,
        with_labels=True,
        node_color=node_colors,
        node_size=2000,
        font_size=10,
        arrowstyle='-|>',
        arrowsize=15
    )
    plt.title("Cleaner Hierarchical Matching Flow (Coarse to Fine)")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    visualize_hierarchical_flow(
        "data/video.mp4",
        frame_idx=10,
        block_coords=(32, 32),
        block_size=16,
        search_window=16,
        pyramid_levels=3
    )
