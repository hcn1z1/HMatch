import argparse
import core.globals

def parse_args():
    program = argparse.ArgumentParser(
        description="Hierarchical Block Matching Program",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100)
    )

    # Define the relevant arguments for this program
    program.add_argument('-a', '--algorithm', type=str, choices=['manual', 'frame', 'hierarchical'], required=True,
                         help="Algorithm to run: manual (single block), frame-wide, or hierarchical visualization.")
    program.add_argument('-v', '--video', type=str, required=True, help="Path to the video file.")
    program.add_argument('-f', '--frame', type=int, default=0, help="Frame index to analyze (default: 0).")
    program.add_argument('-sf', '--secondframe', type=int, default=None, help="Frame index to analyze (default: None).")
    program.add_argument('-b', '--block', type=str, default="32,32",
                         help="Block coordinates (y,x) for manual block matching (default: 32,32).")
    program.add_argument('-m', '--metric', type=str, choices=['time', 'memory'], help="Performance metric to analyze.")
    program.add_argument('-l', '--levels', type=int , help="Pyramid levels to construct",default=3)
    program.add_argument('-g', '--graph', action='store_true', help="Enable graphical output.")
    program.add_argument('-s','--size',type=int , help="Set block size for all block matching (default: 16)\n\n",default=16)

    # Parse arguments
    args = program.parse_args()

    # Update core.globals with the parsed arguments
    core.globals.algorithm = args.algorithm
    core.globals.video = args.video
    core.globals.frame = args.frame
    core.globals.secondframe = args.secondframe
    core.globals.block = tuple(map(int, args.block.split(',')))
    core.globals.metric = args.metric
    core.globals.graph = args.graph
    core.globals.level = args.levels
    core.globals.block_size = args.size