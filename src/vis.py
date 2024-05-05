import os
os.makedirs('./vis', exist_ok=True)
import argparse
from utils import *

def main():
    parser = argparse.ArgumentParser(description='Result Visualization')
    parser.add_argument('--all', type=bool, default=True, help='All sessions in the results directory')
    parser.add_argument('--session_path', type=str, required=False, help='Path for a single session')
    args = parser.parse_args()

    save_dir = './vis'

    if args.all == True:
        session_path_list = get_subdirpath_list()

        # Plot for log info
        log_vis(session_path_list, save_dir, key='train', ylim=[0,100])
        log_vis(session_path_list, save_dir, key='val', ylim=[0,5])

        # Plot for metrics
        metric_vis(session_path_list, 'acc', save_dir)
        metric_vis(session_path_list, 'r2', save_dir)
        metric_vis(session_path_list, 'mse', save_dir)
        metric_vis(session_path_list, 'mae', save_dir)

        # Plot for ground truth vs prediction
        for session_path in session_path_list: 
            pred_vis(session_path, save_dir)
    
    if args.all == False:
        pred_vis(args.session_path, save_dir)

if __name__ == "__main__":
    main()