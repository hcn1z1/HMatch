from scripts.manual_block import run_manual_block
from scripts.frame_block import run_frame_blocks
from scripts.sequence_block import run_sequence_block
from scripts.visualize_hierarchical_flow import visualize_hierarchical_flow
from analysis.profiler import measure_execution_time, measure_memory_usage
from core.argparser import parse_args
import core.globals

def main():
    parse_args()

    print("=== Hierarchical Block Matching ===")
    print(f"Algorithm: {core.globals.algorithm}")
    print(f"Processing Type: {core.globals.processing_type}")
    print(f"Score Algorithm: {core.globals.score_algorithm}")
    print(f"Video: {core.globals.video}")
    print(f"Frame: {core.globals.frame}")
    print(f"Second Frame: {core.globals.secondframe}")
    print(f"Levels: {core.globals.level}")
    print(f"Block Coordinates: {core.globals.block}")
    print(f"Block Size: {core.globals.block_size}")
    print(f"Metric: {core.globals.metric}")
    print(f"Search Window: {core.globals.search_window}")
    print(f"Graph Enabled: {core.globals.graph}")
    print("===================================\n\n")
    if core.globals.processing_type == 'manual':
        print("[INFO] Running manual block matching...")
        run_manual_block(core.globals.video, core.globals.frame, core.globals.block, pyramid_levels=core.globals.level, block_size=core.globals.block_size, second_frame_idx=core.globals.secondframe, plot=core.globals.graph, search_window=core.globals.search_window)

    elif core.globals.processing_type == 'frame':
        print("[INFO] Running frame-wide block matching...")
        run_frame_blocks(core.globals.video, core.globals.frame, pyramid_levels=core.globals.level, block_size=core.globals.block_size, second_frame_idx=core.globals.secondframe, search_window=core.globals.search_window, plot= core.globals.graph)

    elif core.globals.processing_type == 'sequence':
        print("[INFO Running sequenced block matching]")
        run_sequence_block(core.globals.video, core.globals.frame, core.globals.block, pyramid_levels=core.globals.level, block_size=core.globals.block_size, second_frame_idx=core.globals.secondframe, plot=core.globals.graph, search_window=core.globals.search_window)
    
    if core.globals.metric: 
        if core.globals.metric == 'time':
            print("[INFO] Measuring execution time...")
            total_time = measure_execution_time(run_frame_blocks, core.globals.video, core.globals.frame)
            print(f"[RESULT] Total Execution Time: {total_time:.4f} seconds")
        elif core.globals.metric == 'memory':
            print("[INFO] Measuring memory usage...")
            memory_usage, peak_memory = measure_memory_usage(run_frame_blocks, core.globals.video, core.globals.frame)
            print(f"[RESULT] Memory Usage: {memory_usage:.4f} MB, Peak: {peak_memory:.4f} MB")

if __name__ == "__main__":
    main()
