import os
import io
from contextlib import redirect_stdout
import hashlib
import time
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet, vgg, mobilenet, vision_transformer
from baseline_cnn import BaselineCNN
from torchinfo import summary
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from pickle_dataset import PreprocessedDataset

def transform_to_dict(transform):
    # Assuming each transform has a name attribute
    return {t.__class__.__name__: t.__dict__ for t in transform.transforms}

def get_pretrained_model(model_name, num_classes, drop_rate, batch_size, pretrained):
    
    # Dictionary mapping model names to model functions and their respective weights
    model_dict = {
        # SOTA models
        'vgg16': (models.vgg16, vgg.VGG16_Weights.DEFAULT),
        'vgg19': (models.vgg19, vgg.VGG19_Weights.DEFAULT),
        'vgg16_bn': (models.vgg16_bn, vgg.VGG16_BN_Weights.DEFAULT),
        'vgg19_bn': (models.vgg19_bn, vgg.VGG19_BN_Weights.DEFAULT),
        'resnet18': (models.resnet18, resnet.ResNet18_Weights.DEFAULT),
        'resnet34': (models.resnet34, resnet.ResNet34_Weights.DEFAULT),
        'resnet50': (models.resnet50, resnet.ResNet50_Weights.DEFAULT),
        'resnet101': (models.resnet101, resnet.ResNet101_Weights.DEFAULT),
        'resnet152': (models.resnet152, resnet.ResNet152_Weights.DEFAULT),
        'mobilenet_v3_small': (models.mobilenet_v3_small, mobilenet.MobileNet_V3_Small_Weights.DEFAULT),
        'mobilenet_v3_large': (models.mobilenet_v3_large, mobilenet.MobileNet_V3_Large_Weights.DEFAULT),
        'vit_b_16': (models.vit_b_16, vision_transformer.ViT_B_16_Weights.DEFAULT),
        'vit_b_32': (models.vit_b_32, vision_transformer.ViT_B_32_Weights.DEFAULT),
        'vit_l_16': (models.vit_l_16, vision_transformer.ViT_L_16_Weights.DEFAULT),
        'vit_l_32': (models.vit_l_32, vision_transformer.ViT_L_32_Weights.DEFAULT),
        'vit_h_14': (models.vit_h_14, vision_transformer.ViT_H_14_Weights.DEFAULT),

        # Custom models
        'baseline_cnn': (BaselineCNN, None) 
    }

    # Check if the model name is in the dictionary
    if model_name not in model_dict:
        raise ValueError(f"Model name '{model_name}' is not supported. Please choose from {list(model_dict.keys())}.")

    model = None
    
    if model_name in model_dict:
        model, weights = model_dict[model_name]
        
        if pretrained:
            model = model(weights=weights)
        else:
            try:
                model = model()
            except:
                model = model(drop_rate, num_classes)
        
        if model_name.startswith('baseline'):
            pass
        elif model_name.startswith('vit_'):
            num_features = model.heads.head.in_features
            model.heads.head = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(256, num_classes)
                    )
        else:
            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
                # Models use 'classifier' attribute
                num_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(256, num_classes)
                    )
            elif hasattr(model, 'fc') and isinstance(model.fc, nn.Module):
                # Models use 'fc' attribute for the fully connected layer
                num_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(256, num_classes)
                    )
    
    # Create a StringIO stream to capture the output
    f = io.StringIO()
    with redirect_stdout(f):
        summary(model, input_size=(batch_size, 3, 224, 224))
    
    # Get the summary from the StringIO stream
    model_summary = f.getvalue()

    return model, model_summary

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def generate_unique_hash():
    # Loop until a unique hash is generated
    while True:
        unique_hash = hashlib.md5(str(time.time()).encode()).hexdigest()
        session_results_dir = os.path.join('./results', unique_hash)
        if not os.path.exists(session_results_dir):
            break
        time.sleep(0.1)  # Wait a bit before trying a new hash
    return session_results_dir

def list_subdirectories(directory_path):
    subdirectories = [entry for entry in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, entry))]
    return subdirectories

def json2dict(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

def get_subdirpath_list(directory_path='results'):
    result_path_list = []
    subdirs = list_subdirectories(directory_path)
    for subdir in subdirs:
        cur_path = os.path.join('results', subdir)
        result_path_list.append(cur_path)
    return result_path_list

def preprocess_and_save_dataset(dataloader, save_path):
    if os.path.exists(save_path):
        print(f"Preprocessed data file '{save_path}' already exists. Continuing without reprocessing.")
        return

    preprocessed_data = []
    for images, labels in tqdm(dataloader, desc='Progress'):
        preprocessed_data.append((images, labels))

    with open(save_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)
        print(f"Preprocessed data saved to '{save_path}'.")

def load_preprocessed_dataset(pkl_path):
    with open(pkl_path, 'rb') as f:
        preprocessed_data = pickle.load(f)
    return preprocessed_data

def create_dataset_from_preprocessed(pkl_path, transform):
    preprocessed_data = load_preprocessed_dataset(pkl_path)
    preprocessed_dataset = PreprocessedDataset(preprocessed_data, transform=transform)
    return preprocessed_dataset

def create_dataloader_from_preprocessed(pkl_path, batch_size, transform, shuffle=True):
    preprocessed_dataset = create_dataset_from_preprocessed(pkl_path, transform)
    preprocessed_dataloader = DataLoader(preprocessed_dataset, batch_size=batch_size, shuffle=shuffle)
    return preprocessed_dataloader






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

def pred_vis(session_path, save_dir, save=True):
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
    if save == True:
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
    
    plt.ylabel(metric.upper(), fontsize=15)
    if ylim != None:
        plt.ylim(ylim)
    plt.title(f'Comparison of Models Based on {metric.upper()}', fontsize=20)
    plt.xticks(rotation=45, fontsize=20)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{metric}_plot.png'))
