import os
os.makedirs('./vis', exist_ok=True)
import argparse
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score

def log_vis(*arg, save_dir, ylim=[0, 100], key='test', epochs=100, gaussian_smooth=True, sigma=4):
    plt.figure(figsize=(5,5))

    for path_ in arg:
        cur_info_path = os.path.join(path_, 'info.json')

        if key == 'test':
            cur_log_path = os.path.join(path_, 'test_log.json')
        else: 
            cur_log_path = os.path.join(path_, 'train_log.json')

        log_curve = json2dict(cur_log_path)[f'{key}_loss'][0:epochs]
        log_label = json2dict(cur_info_path)['model_name']

        # Apply Gaussian smoothing
        if gaussian_smooth: 
            log_curve = gaussian_filter1d(log_curve, sigma=sigma)
        plt.plot(log_curve, label=log_label) 

    plt.title(f'{key.upper()} LOSS')
    plt.ylim(ylim)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{key}_loss_plot.png'))

def pred_vis(session_path, save_dir):
    pred_info_path = os.path.join(session_path, 'pred.json')
    model_info_path = os.path.join(session_path, 'info.json')
    test_log_path = os.path.join(session_path, 'test_log.json')

    pred_info = json2dict(pred_info_path)
    model_info = json2dict(model_info_path)
    test_log =  json2dict(test_log_path)
    model_name = model_info['model_name']
    test_loss = test_log['test_loss'][0] 

    gt = np.array(pred_info['true_labels']).reshape(1, -1)
    pred = np.array(pred_info['predictions']).reshape(1, -1)

    r_squared = r2_score(gt.reshape(-1, 1), pred.reshape(-1, 1))
    accuracy = np.mean(np.abs(gt - pred) <= 1)
    print(f'R^2 Score: {r_squared:.3f}')
    print(f'Accuracy within ±1 unit: {accuracy:.3f}')
    print(f'Loss: {test_loss:.3f}')

    # Plot
    plt.figure(figsize=(5,5))
    plt.scatter(gt, pred, c='pink', alpha=0.3)
    plt.title(f'{model_name}')

    # Create a line space for the ideal line where ground truth equals predictions
    line_space = np.linspace(min(np.minimum(gt, pred)), max(np.maximum(gt, pred)), 30)
    plt.plot(line_space, line_space, 'r', alpha=0.3, label='Ideal: y = x')  # Red dashed line
    plt.plot(line_space, line_space + 1, 'b', alpha=0.3, label='Error Margin +1')  # Blue dashed line for +1 error margin
    plt.plot(line_space, line_space - 1, 'b', alpha=0.3, label='Error Margin -1')  # Blue dashed line for -1 error margin

    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{model_name}_gt_pred_plot.png'))

def main():
    parser = argparse.ArgumentParser(description='Result Visualization')
    parser.add_argument('--session_path', type=str, required=False, help='Path for a single session')
    parser.add_argument('--all', type=bool, default=True, help='All sessions in the results directory')
    args = parser.parse_args()

    if args.all == True:
        result_path_list = get_subdirpath_list()
        log_vis(*result_path_list, key='train')
        log_vis(*result_path_list, key='val')
        log_vis(*result_path_list, key='test')








if __name__ == "__main__":
    main()