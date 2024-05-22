import os
import io
import argparse
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
    """
    Convert a torchvision.transforms.Compose object's transforms to a dictionary.

    Args:
        transform (torchvision.transforms.Compose): A Compose object containing multiple transforms.

    Returns:
        dict: A dictionary with transform class names as keys and their __dict__ attributes as values.
    """
    # Assuming each transform has a name attribute
    return {t.__class__.__name__: t.__dict__ for t in transform.transforms}

def get_pretrained_model(model_name, num_classes, drop_rate, batch_size, pretrained):
    """
    Retrieve a pretrained model with a custom classifier head based on the specified parameters.

    Args:
        model_name (str): Name of the model to retrieve.
        num_classes (int): Number of classes for the final output layer.
        drop_rate (float): Dropout rate for the classifier head.
        batch_size (int): Batch size for the input tensor.
        pretrained (bool): Flag to use pretrained weights or not.

    Raises:
        ValueError: If the model_name is not supported.

    Returns:
        tuple: A tuple containing the model and its summary.
    """
    # Dictionary mapping model names to model functions and their respective weights
    # SOTA models and custom models are included
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

    # Check if the model name is in the dictionary and raise an error if not
    if model_name not in model_dict:
        raise ValueError(f"Model name '{model_name}' is not supported. Please choose from {list(model_dict.keys())}.")

    # Initialize the model variable
    model = None
    
    # Retrieve the model and its weights if the model name is in the dictionary
    if model_name in model_dict:
        model, weights = model_dict[model_name]
        
        # Load the model with pretrained weights if specified
        if pretrained:
            model = model(weights=weights)
        else:
            # Attempt to instantiate the model without weights
            try:
                model = model()
            except:
                # If the model requires additional parameters, provide them
                model = model(drop_rate, num_classes)
        
        # Customize the classifier head based on the model type
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
    
    # Create a StringIO stream to capture the output of the summary function
    f = io.StringIO()
    with redirect_stdout(f):
        summary(model, input_size=(batch_size, 3, 224, 224))
    
    # Retrieve the captured summary from the StringIO stream
    model_summary = f.getvalue()

    # Return the model and its summary
    return model, model_summary

def count_parameters(model):
    """
    Count the total number of parameters in a model.

    Args:
        model (torch.nn.Module): The model to count parameters for.

    Returns:
        int: The total number of parameters in the model.
    """
    # Calculate the total number of parameters using a generator expression
    return sum(p.numel() for p in model.parameters())

def generate_unique_hash():
    """
    Generate a unique hash to be used as a directory name for session results.

    Returns:
        str: The path to the unique session results directory.
    """
    # Loop until a unique hash is generated
    while True:
        # Generate a hash based on the current time
        unique_hash = hashlib.md5(str(time.time()).encode()).hexdigest()
        # Construct the directory path using the unique hash
        session_results_dir = os.path.join('./results', unique_hash)
        # Check if the directory already exists
        if not os.path.exists(session_results_dir):
            break
        # Wait a bit before trying to generate a new hash
        time.sleep(0.1)
    # Return the unique session results directory path
    return session_results_dir

def list_subdirectories(directory_path):
    """
    List all subdirectories within a given directory path.

    Args:
        directory_path (str): The path to the directory.

    Returns:
        list: A list of subdirectory names within the given directory path.
    """
    # Use a list comprehension to filter for directories only
    subdirectories = [entry for entry in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, entry))]
    # Return the list of subdirectories
    return subdirectories

def json2dict(json_file_path):
    """
    Load and return the contents of a JSON file as a dictionary.

    Parameters:
    - json_file_path (str): The file path to the JSON file.

    Returns:
    - dict: The contents of the JSON file.
    """
    with open(json_file_path, 'r') as file:
        # Load the JSON file and return its contents
        return json.load(file)

def get_subdirpath_list(directory_path='results'):
    """
    Generate a list of subdirectory paths within a given directory.

    Parameters:
    - directory_path (str): The parent directory path. Defaults to 'results'.

    Returns:
    - list: A list of subdirectory paths.
    """
    result_path_list = []
    # Retrieve a list of subdirectories
    subdirs = list_subdirectories(directory_path)
    for subdir in subdirs:
        # Join the parent directory with the subdirectory
        cur_path = os.path.join('results', subdir)
        result_path_list.append(cur_path)
    return result_path_list

def preprocess_and_save_dataset(dataloader, save_path):
    """
    Preprocess data from a dataloader and save it to a file.

    Parameters:
    - dataloader (DataLoader): The dataloader containing the dataset to preprocess.
    - save_path (str): The file path to save the preprocessed data.
    """
    if os.path.exists(save_path):
        # If the file already exists, print a message and return
        print(f"Preprocessed data file '{save_path}' already exists. Continuing without reprocessing.")
        return

    preprocessed_data = []
    # Process each batch of images and labels
    for images, labels in tqdm(dataloader, desc='Progress'):
        preprocessed_data.append((images, labels))

    with open(save_path, 'wb') as f:
        # Save the preprocessed data to a file
        pickle.dump(preprocessed_data, f)
        print(f"Preprocessed data saved to '{save_path}'.")

def load_preprocessed_dataset(pkl_path):
    """
    Load a preprocessed dataset from a pickle file.

    Parameters:
    pkl_path (str): The path to the pickle file containing the preprocessed data.

    Returns:
    preprocessed_data: The data loaded from the pickle file.
    """
    # Open the pickle file in read-binary mode
    with open(pkl_path, 'rb') as f:
        # Load the data from the pickle file
        preprocessed_data = pickle.load(f)
    return preprocessed_data

def create_dataset_from_preprocessed(pkl_path, transform):
    """
    Create a dataset object from preprocessed data with optional transforms.

    Parameters:
    pkl_path (str): The path to the pickle file containing the preprocessed data.
    transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.

    Returns:
    PreprocessedDataset: The dataset object created from the preprocessed data.
    """
    # Load the preprocessed data from the pickle file
    preprocessed_data = load_preprocessed_dataset(pkl_path)
    # Create a dataset object with the preprocessed data and the transform
    preprocessed_dataset = PreprocessedDataset(preprocessed_data, transform=transform)
    return preprocessed_dataset

def create_dataloader_from_preprocessed(pkl_path, batch_size, transform, shuffle=True):
    """
    Create a DataLoader from preprocessed data with optional transforms and shuffling.

    Parameters:
    pkl_path (str): The path to the pickle file containing the preprocessed data.
    batch_size (int): How many samples per batch to load.
    transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
    shuffle (bool, optional): Set to True to have the data reshuffled at every epoch (default: True).

    Returns:
    DataLoader: The DataLoader object for the preprocessed dataset.
    """
    # Create a dataset object from the preprocessed data with the given transform
    preprocessed_dataset = create_dataset_from_preprocessed(pkl_path, transform)
    # Create a DataLoader with the dataset, batch size, and shuffle option
    preprocessed_dataloader = DataLoader(preprocessed_dataset, batch_size=batch_size, shuffle=shuffle)
    return preprocessed_dataloader

def log_vis(session_path_list, save_dir, ylim=[0, 100], key='val', epochs=100, gaussian_smooth=True, sigma=3):
    """
    Visualize and save a plot of the loss curves from training sessions.

    Parameters:
    session_path_list (list of str): List of paths to session directories containing 'info.json' and 'train_log.json'.
    save_dir (str): Directory where the plot image will be saved.
    ylim (list of int): The y-axis limits for the plot.
    key (str): The key prefix for loss values in 'train_log.json'.
    epochs (int): Number of epochs to plot.
    gaussian_smooth (bool): Apply Gaussian smoothing to the loss curve if True.
    sigma (int): The standard deviation for Gaussian kernel.

    Returns:
    None: The function saves the plot image to the specified directory.
    """
    # Initialize a new figure with specified size
    plt.figure(figsize=(5,5))

    # Loop through each session path
    for path_ in session_path_list:
        # Construct paths to the info and log JSON files
        cur_info_path = os.path.join(path_, 'info.json')
        cur_log_path = os.path.join(path_, 'train_log.json')

        # Extract the loss curve and model name from the JSON files
        log_curve = json2dict(cur_log_path)[f'{key}_loss'][0:epochs]
        log_label = json2dict(cur_info_path)['model_name']

        # Apply Gaussian smoothing if enabled
        if gaussian_smooth: 
            log_curve = gaussian_filter1d(log_curve, sigma=sigma)
        # Plot the smoothed curve with the corresponding label
        plt.plot(log_curve, label=log_label) 

    # Set the title, axis labels, and y-axis limits for the plot
    plt.title(f'{key.upper()} LOSS')
    plt.ylim(ylim)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # Adjust layout for better fit and add legend
    plt.tight_layout()
    plt.legend()

    # Save the plot to the specified directory
    plt.savefig(os.path.join(save_dir, f'{key}_loss_plot.png'))

def pred_vis(session_path, save_dir, save=True):
    """
    Visualize the ground truth vs predictions scatter plot.

    Parameters:
    session_path (str): The directory path where prediction and model info are stored.
    save_dir (str): The directory path where the plot image will be saved.
    save (bool): A flag to determine whether to save the plot image or not.
    """
    # Construct paths to the prediction and model information JSON files
    pred_info_path = os.path.join(session_path, 'pred.json')
    model_info_path = os.path.join(session_path, 'info.json')

    # Load the prediction and model information into dictionaries
    pred_info = json2dict(pred_info_path)
    model_info = json2dict(model_info_path)
    model_name = model_info['model_name']

    # Convert the true labels and predictions into numpy arrays and reshape for plotting
    gt = np.array(pred_info['true_labels']).reshape(1, -1)[0]
    pred = np.array(pred_info['predictions']).reshape(1, -1)[0]

    # Initialize the plot with specified figure size
    plt.figure(figsize=(5,5))
    # Create a scatter plot of ground truth vs predictions
    plt.scatter(gt, pred, c='pink', alpha=0.3)

    # Generate a line space for the ideal line where ground truth equals predictions
    line_space = np.linspace(min(gt), max(gt), 100)
    # Plot the ideal line where predictions are exactly equal to the ground truth
    plt.plot(line_space, line_space, 'r', alpha=1, label='Ideal: pred = truth')
    # Plot the error margin lines at +1 and -1 cm
    plt.plot(line_space, line_space + 1, 'b', alpha=1, label='Error Margin ±1 (cm)')
    plt.plot(line_space, line_space - 1, 'b', alpha=1)

    # Label the x-axis and y-axis
    plt.xlabel('Ground Truth (cm)')
    plt.ylabel('Prediction (cm)')
    # Set the title of the plot using the model name
    plt.title(f'{model_name} Testset Ground Truth vs Prediction')
    # Display the legend
    # plt.legend()
    # Adjust the layout to ensure everything fits without overlap
    plt.tight_layout()

    # Save the plot if the save flag is True
    if save:
        plt.savefig(os.path.join(save_dir, f'{model_name}_gt_pred_plot.png'))

def metric_vis(session_path_list, metric, save_dir, ylim=None):
    """
    Visualize the comparison of different models based on a specified metric.

    Parameters:
    session_path_list (list): A list of directory paths for different model sessions.
    metric (str): The metric to compare ('r2', 'acc', 'mse', 'mae').
    save_dir (str): The directory path where the bar plot image will be saved.
    ylim (tuple): Optional. The y-axis limits for the plot.
    """
    # Initialize lists to store paths and metrics
    pred_info_path_list = []
    model_info_path_list = []
    model_name_list = []
    metric_list = []

    # Loop through each session path to collect model names and calculate metrics
    for session_path in session_path_list:
        # Construct paths to the prediction and model information JSON files
        pred_info_path = os.path.join(session_path, 'pred.json')
        model_info_path = os.path.join(session_path, 'info.json')
        pred_info_path_list.append(pred_info_path)
        model_info_path_list.append(model_info_path)

        # Load the model name from the model information file
        model_name_list.append(json2dict(model_info_path)['model_name'])

    # Calculate the specified metric for each model
    for pred_info_path in pred_info_path_list:
        pred_info = json2dict(pred_info_path)
        gt = np.array(pred_info['true_labels']).reshape(1, -1)
        pred = np.array(pred_info['predictions']).reshape(1, -1)

        # Calculate the R-squared score
        if metric == 'r2':
            r2 = r2_score(gt.reshape(-1, 1), pred.reshape(-1, 1))
            metric_list.append(r2)
        # Calculate the accuracy within an error margin of 1 cm
        elif metric == 'acc':
            accuracy = np.mean(np.abs(gt - pred) <= 1)
            metric_list.append(accuracy)
        # Calculate the mean squared error
        elif metric == 'mse':
            mse = mean_squared_error(gt, pred)
            metric_list.append(mse)
        # Calculate the mean absolute error
        elif metric == 'mae':
            mae = mean_absolute_error(gt, pred)
            metric_list.append(mae)
        else:
            print('Metric not yet supported.')
            return

    # Sort the models based on the calculated metric values
    sorted_indices = np.argsort(metric_list)[::-1]
    sorted_model_names = np.array(model_name_list)[sorted_indices]
    sorted_metrics = np.array(metric_list)[sorted_indices]

    # Set the color scheme based on the type of metric
    if metric in ['acc', 'r2']:
        # Higher is better
        colors = ['crimson' if i == 0 else 'skyblue' for i in range(len(sorted_model_names))]
    elif metric in ['mse', 'mae']:
        # Lower is better
        colors = ['crimson' if i == len(sorted_model_names) - 1 else 'skyblue' for i in range(len(sorted_model_names))]
    else:
        # Default color scheme if metric type is unknown
        colors = ['skyblue' for _ in range(len(sorted_model_names))]

    # Initialize the bar plot with specified figure size
    plt.figure(figsize=(15, 5))
    # Create a bar plot to compare the models based on the metric
    bars = plt.bar(sorted_model_names, sorted_metrics, color=colors, width=0.7)

    # Highlight the best model and show its score
    best_model_index = 0 if metric in ['r2', 'acc'] else len(session_path_list) - 1
    bars[best_model_index].set_label('Best Model')
    plt.legend(fontsize=15)

    # Annotate the best model's bar with its score
    plt.text(best_model_index, sorted_metrics[best_model_index], f'{sorted_metrics[best_model_index]:.2f}', 
             ha='center', va='bottom', fontsize=20)

    # Label the y-axis with the metric name
    plt.ylabel(metric.upper(), fontsize=15)
    # Set the y-axis limits if provided
    if ylim is not None:
        plt.ylim(ylim)
    # Set the title of the plot
    plt.title(f'Comparison of Models Based on {metric.upper()}', fontsize=20)
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, fontsize=20)
    # Adjust the layout to ensure everything fits without overlap
    plt.tight_layout()

    # Save the plot image
    plt.savefig(os.path.join(save_dir, f'{metric}_plot.png'))

def worst_pred(session_path, dataset, idxes, mode):
    """
    Visualize the samples with the worst predictions compared to ground truth.

    This function loads the prediction and ground truth data from a JSON file, 
    calculates the difference between predictions and ground truth based on the
    specified mode, finds the samples with the largest errors, and plots the 
    ground truth vs prediction scatter plot along with the corresponding images.

    Args:
        session_path (str): Path to the session directory containing the 'pred.json' file.
        dataset (Dataset): The dataset object containing the images and labels.
        idxes (list): The indexes of the target samples to visualize, sorted by worst predictions.
        mode (str): The mode for calculating prediction errors. Options are:
            'abs' - Absolute difference between ground truth and prediction
            'h'   - Positive difference (prediction higher than ground truth) 
            'l'   - Negative difference (prediction lower than ground truth)

    Returns:
        None

    Raises:
        AssertionError: If loaded ground truth data does not match the dataset labels
                        within the specified tolerance.

    Example:
        worst_pred('path/to/session', my_dataset, [0, 1, 2, 3, 4], 'abs')
    """
    
    # Load prediction and ground truth data from JSON file
    pred_path = os.path.join(session_path, 'pred.json')
    pred_info = json2dict(pred_path)
    
    # Extract ground truth and prediction arrays
    gt = np.array(pred_info['true_labels']).reshape(1, -1)[0]  
    pred = np.array(pred_info['predictions']).reshape(1, -1)[0]

    # Calculate prediction errors based on specified mode
    if mode == 'abs':
        # Absolute difference between ground truth and prediction
        diff = np.abs(gt - pred)
    elif mode == 'h':  
        # Positive difference (prediction higher than ground truth)
        diff = pred - gt
    elif mode == 'l':
        # Negative difference (prediction lower than ground truth)
        diff = gt - pred 
    else:
        print('Warning: Mode argument is not recognized.')
        return

    # Sanity check: Ensure loaded ground truth data matches dataset labels within tolerance
    assert np.allclose(gt.round(4), np.array(dataset.labels).reshape(1,-1)[0].round(4), 
                       rtol=1e-4), "Loaded data does not match dataset labels"

    # Find sample indexes with largest errors
    sorted_idxes = np.argsort(diff)[::-1]
    target_idxes = np.take(sorted_idxes, idxes)

    # Plot results for each target sample
    for idx in target_idxes:
        # Extract image data and transpose dimensions (CWH to WHC)
        img_cwh = np.array(dataset.images[idx][0]) 
        img_whc = img_cwh.transpose(1, 2, 0)

        # Create a new figure with specified size
        plt.figure(figsize=(7.5, 4))

        # Plot scatter plot of ground truth vs prediction
        plt.subplot(1, 2, 1)
        plt.scatter(gt, pred, c='pink', alpha=0.3)
        plt.scatter(gt[idx], pred[idx], c='red', alpha=1, label='target')  
        plt.title(f'Ground Truth vs Prediction')
        line_space = np.linspace(58, 88, 30)
        plt.plot(line_space, line_space, 'r', alpha=1, label='Ideal: pred = truth')
        plt.plot(line_space, line_space + 1, 'b', alpha=1, label='Error Margin +-1 (cm)') 
        plt.plot(line_space, line_space - 1, 'b', alpha=1)
        plt.xlabel('Ground Truth (cm)')
        plt.ylabel('Prediction (cm)')
        plt.legend()

        # Plot the corresponding image
        plt.subplot(1, 2, 2)
        plt.imshow(img_whc) 
        plt.title(f"GT: {gt[idx]:.3f}, Pred: {pred[idx]:.3f}, Diff: {diff[idx]:.3f}")
        plt.axis('off')

        # Adjust the layout and display the plot
        plt.tight_layout()
        plt.show()

def validate_and_prompt_path(path):
    """
    Recursively validate the provided path and prompt for a new one if invalid.

    Parameters:
    path (str): The file system path to be validated.

    Returns:
    str: A valid file system path.
    """
    # Check if the provided path exists in the file system
    if not os.path.exists(path):
        # Inform the user that the provided path is invalid
        print("The provided path is invalid. Please enter a valid path.")
        # Prompt the user to enter a new path
        new_path = input("Enter a new path: ")
        # Recursively call the function with the new path
        return validate_and_prompt_path(new_path)
    else:
        # Confirm to the user that the provided path is valid
        print("The provided path is valid.")
        # Return the valid path
        return path

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')