import os
os.makedirs('./vis', exist_ok=True)
import argparse
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def log_vis(session_path_list, save_dir, ylim=[0, 100], key='val', epochs=10, gaussian_smooth=True, sigma=3):
    plt.figure(figsize=(5,5))

    for path_ in session_path_list:
        cur_info_path = os.path.join(path_, 'info.json')
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

    pred_info = json2dict(pred_info_path)
    model_info = json2dict(model_info_path)
    model_name = model_info['model_name']

    gt = np.array(pred_info['true_labels']).reshape(1, -1)
    pred = np.array(pred_info['predictions']).reshape(1, -1)

    # Plot
    plt.figure(figsize=(5,5))
    plt.scatter(gt, pred, c='pink', alpha=0.3)
    plt.title(f'{model_name} Testset Ground Truth vs Prediction')

    # Create a line space for the ideal line where ground truth equals predictions
    line_space = np.linspace(58, 88, 30)
    plt.plot(line_space, line_space, 'r', alpha=1, label='Ideal: pred = truth')  # Red dashed line
    plt.plot(line_space, line_space + 1, 'b', alpha=1, label='Error Margin +-1')  # Blue dashed line for +1 error margin
    plt.plot(line_space, line_space - 1, 'b', alpha=1)  # Blue dashed line for -1 error margin

    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{model_name}_gt_pred_plot.png'))

def metric_vis(session_path_list, metric, save_dir, ylim=None):
    pred_info_path_list = []
    model_info_path_list = []
    for session_path in session_path_list:
        pred_info_path = os.path.join(session_path, 'pred.json')
        model_info_path = os.path.join(session_path, 'info.json')
        pred_info_path_list.append(pred_info_path)
        model_info_path_list.append(model_info_path)
    
    model_name_list = []
    metric_list = []
    for i in model_info_path_list:
        model_name_list.append(json2dict(i)['model_name'])
    for i in pred_info_path_list:
        pred_info = json2dict(i)
        gt = np.array(pred_info['true_labels']).reshape(1, -1)
        pred = np.array(pred_info['predictions']).reshape(1, -1)
        if metric == 'r2':
            r2 = r2_score(gt.reshape(-1, 1), pred.reshape(-1, 1))
            metric_list.append(r2)
        elif metric == 'acc':
            accuracy = np.mean(np.abs(gt - pred) <= 1)
            metric_list.append(accuracy)
        elif metric == 'mse':
            mse = mean_squared_error(gt, pred)
            metric_list.append(mse)
        elif metric == 'mae':
            mae = mean_absolute_error(gt, pred)
            metric_list.append(mae)
        else:
            print('Metric not yet supported.')
            return
    
    # Sort the metrics and model names based on the metric values
    sorted_indices = np.argsort(metric_list)[::-1]
    sorted_model_names = np.array(model_name_list)[sorted_indices]
    sorted_metrics = np.array(metric_list)[sorted_indices]
    
    # Color settings
    if metric == 'r2' or metric == 'acc':
        colors = ['crimson' if i == 0 else 'skyblue' for i in range(len(sorted_model_names))]
    elif metric == 'mse' or metric == 'mae':
        colors = ['crimson' if i == len(session_path_list)-1 else 'skyblue' for i in range(len(sorted_model_names))]
    
    # Create the bar plot
    plt.figure(figsize=(15, 5))
    bars = plt.bar(sorted_model_names, sorted_metrics, color=colors, width=0.7)
    
    # Highlight the best model and show its score
    best_model_index = 0 if metric in ['r2', 'acc'] else len(session_path_list) - 1
    bars[best_model_index].set_label('Best Model')
    plt.legend(fontsize=15)

    # Annotate the best model's bar with its score
    plt.text(best_model_index, sorted_metrics[best_model_index], f'{sorted_metrics[best_model_index]:.2f}', 
            ha='center', va='bottom', fontsize=20)
    
    # plt.xlabel('Model Names')
    plt.ylabel(metric.upper(), fontsize=15)
    if ylim != None:
        plt.ylim(ylim)
    plt.title(f'Comparison of Models Based on {metric.upper()}', fontsize=20)
    plt.xticks(rotation=45, fontsize=20)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{metric}_plot.png'))

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