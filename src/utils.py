import os
import hashlib
import time
import json
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet, vgg
from torchsummary import summary
from torch.utils.data import DataLoader
from pickle_dataset import PreprocessedDataset

def transform_to_dict(transform):
    # Assuming each transform has a name attribute
    return {t.__class__.__name__: t.__dict__ for t in transform.transforms}

def get_pretrained_model(model_name, num_classes, drop_rate, device, pretrained=True, print_summary=False):
    
    # Dictionary mapping model names to model functions and their respective weights
    model_dict = {
        'vgg16': (models.vgg16, vgg.VGG16_Weights.DEFAULT),
        'vgg19': (models.vgg19, vgg.VGG19_Weights.DEFAULT),
        'vgg16_bn': (models.vgg16_bn, vgg.VGG16_BN_Weights.DEFAULT),
        'vgg19_bn': (models.vgg19_bn, vgg.VGG19_BN_Weights.DEFAULT),
        'resnet18': (models.resnet18, resnet.ResNet18_Weights.DEFAULT),
        'resnet34': (models.resnet34, resnet.ResNet34_Weights.DEFAULT),
        'resnet50': (models.resnet50, resnet.ResNet50_Weights.DEFAULT),
        'resnet101': (models.resnet101, resnet.ResNet101_Weights.DEFAULT),
        'resnet152': (models.resnet152, resnet.ResNet152_Weights.DEFAULT)
    }

    # Check if the model name is in the dictionary
    if model_name not in model_dict:
        raise ValueError(f"Model name '{model_name}' is not supported. Please choose from {list(model_dict.keys())}.")

    model = None
    input_size = (3, 224, 224)  # Default input size for VGG and ResNet models
    
    if model_name in model_dict:
        model_func, weights = model_dict[model_name]
        if pretrained:
            model = model_func(weights=weights)
        else:
            model = model_func(weights=None)
        
        if model_name.startswith('vgg'):
            # VGG models use 'classifier' attribute
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
        elif model_name.startswith('res'):
            # ResNet models use 'fc' attribute
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
        else:
            raise ValueError("Model name must start with 'vgg' or 'res'.")
        
        if print_summary:
            summary(model.to(torch.device(device)), input_size=input_size)

    return model

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

def create_dataloader_from_preprocessed(pkl_path, batch_size, shuffle=True):
    preprocessed_data = load_preprocessed_dataset(pkl_path)
    preprocessed_dataset = PreprocessedDataset(preprocessed_data)
    preprocessed_dataloader = DataLoader(preprocessed_dataset, batch_size=batch_size, shuffle=shuffle)
    return preprocessed_dataloader