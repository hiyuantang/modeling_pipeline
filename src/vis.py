import os
os.makedirs('./vis', exist_ok=True)
import argparse
from utils import *

def main():
    """
    The main function to handle result visualization based on user arguments.

    It supports visualizing all sessions within a results directory or a single session.
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Result Visualization')
    # Add an argument to determine whether to process all sessions
    parser.add_argument('--all', type=bool, default=True, help='All sessions in the results directory')
    # Add an argument to specify the path for a single session
    parser.add_argument('--session_path', type=str, required=False, help='Path for a single session')
    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Define the directory where visualizations will be saved
    save_dir = './vis'

    # Check if the user wants to process all sessions
    if args.all:
        # Obtain a list of subdirectory paths for all sessions
        session_path_list = get_subdirpath_list()

        # Visualize log information for training and validation
        log_vis(session_path_list, save_dir, key='train', ylim=[0,100])
        log_vis(session_path_list, save_dir, key='val', ylim=[0,5])

        # Visualize metrics: accuracy, R-squared, mean squared error, and mean absolute error
        metric_vis(session_path_list, 'acc', save_dir)
        metric_vis(session_path_list, 'r2', save_dir)
        metric_vis(session_path_list, 'mse', save_dir)
        metric_vis(session_path_list, 'mae', save_dir)

        # Visualize ground truth vs prediction for each session
        for session_path in session_path_list: 
            pred_vis(session_path, save_dir)
    
    # If the user wants to process a single session
    elif not args.all:
        # Visualize ground truth vs prediction for the specified session
        pred_vis(args.session_path, save_dir)

# Entry point of the script
if __name__ == "__main__":
    main()