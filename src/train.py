import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import json
import copy
from torch.utils.data import random_split
from utils import *

def train(model_name, train_data_dir, epochs, batch_size, learning_rate, drop_rate, pre_trained_torchvision, pre_trained_session_dir, 
          device, save_interval, patience, train_split, session_dir, update):  
    """
    Train a deep learning model with the given parameters.

    Args:
        model_name (str): Name of the model to be used.
        train_data_dir (str): Directory path to the training data.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Size of the batches for training and validation.
        learning_rate (float): Learning rate for the optimizer.
        drop_rate (float): Dropout rate for the model.
        pre_trained_torchvision (bool): Flag to use pretrained weights from torchvision.
        pre_trained_session_dir (bool): Flag to use pretrained weights from session.
        device (str): Device to run the training on ('cpu' or 'cuda').
        save_interval (int): Interval of batches after which the model state is saved.
        patience (int): Number of epochs to wait for improvement before early stopping.
        train_split (float): Fraction of data to be used for training.
        session_dir (str): Directory path to save the model and logs.
        update (str): Transfer Learning updating all parameters or only the readout.

    Returns:
        None
    """

    # Define transformations for the training dataset
    train_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),  
    # transforms.RandomRotation(15), 
    # transforms.Normalize(mean=[0.3568, 0.3568, 0.3568], std=[0.3512, 0.3512, 0.3512]), # Means and Standard Deviations for depth maps
    # transforms.Normalize(mean=[0.2341, 0.2244, 0.2061], std=[0.1645, 0.1472, 0.1261]), # Means and Standard Deviations for RGB images
    transforms.Normalize(mean=[0.3262, 0.3042, 0.2858], std=[0.3102, 0.2950, 0.2840]), # Means and Standard Deviations for Kaggle images
    ])

    # Define transformations for the validation dataset
    val_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),   
    # transforms.Normalize(mean=[0.3568, 0.3568, 0.3568], std=[0.3512, 0.3512, 0.3512]), # Means and Standard Deviations for depth maps
    # transforms.Normalize(mean=[0.2341, 0.2244, 0.2061], std=[0.1645, 0.1472, 0.1261]), # Means and Standard Deviations for RGB images
    transforms.Normalize(mean=[0.3262, 0.3042, 0.2858], std=[0.3102, 0.2950, 0.2840]), # Means and Standard Deviations for Kaggle images
    ])

    # Load and preprocess the dataset
    print('...loading training dataset')
    dataset = create_dataset_from_preprocessed(train_data_dir, None)

    # Split dataset into training and validation sets
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply transformations to the training and validation datasets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create DataLoader objects for the training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    print('...training dataset loading completed')

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # Initialize a dictionary to log the loss values
    loss_log = {'train_loss': [], 'val_loss': []}

    # Load the model with the specified parameters
    model, _ = get_pretrained_model(model_name, num_classes=1, drop_rate=drop_rate, batch_size=batch_size, pretrained=pre_trained_torchvision)
    
    # Check if there is a pre-trained session directory
    if pre_trained_session_dir is not None:
        pre_trained_model_weights_path = os.path.join(pre_trained_session_dir, 'best_model.pth')
        
        # Load the state dictionary from the .pth file
        state_dict = torch.load(pre_trained_model_weights_path)
        
        # Load the state dictionary into the model
        model.load_state_dict(state_dict)

    device = torch.device(device)
    model = model.to(device)

    # Set parameter updates
    if update == 'all':
        for param in model.parameters():
            param.requires_grad = True
    elif update == 'readout':
        for param in model.parameters():
            param.requires_grad = False
        if model_name.startswith('baseline'):
            for param in model.fc_sq.parameters():
                param.requires_grad = True
        elif model_name.startswith('vit_'):
            for param in model.heads.head.parameters():
                param.requires_grad = True
        else:
            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'fc') and isinstance(model.fc, nn.Module):
                for param in model.fc.parameters():
                    param.requires_grad = True

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()

    # Filter parameters to include only those with requires_grad=True
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Define the optimizer
    optimizer = optim.Adam(trainable_params, lr=learning_rate)

    # Start the training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')):
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Save the model state at specified intervals
            if save_interval == -1:
                continue
            else:
                if (i + 1) % save_interval == 0:
                    model_save_path = os.path.join(session_dir, f'{model_name}_step_{i+1}.pth')
                    torch.save(model.state_dict(), model_save_path)

        # Calculate the average training loss for the epoch
        epoch_loss = running_loss / len(train_loader)

        # Start the validation loop
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Log both training and validation losses
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{epochs} - Training MSELoss: {epoch_loss:.4f} | Validation MSELoss: {val_loss:.4f}')
        loss_log['train_loss'].append(epoch_loss)
        loss_log['val_loss'].append(val_loss)

        # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {patience} epochs without improvement.')
                break
    # Save the best model state if it exists
    if best_model_state is not None:
        best_model_save_path = os.path.join(session_dir, 'best_model.pth')
        torch.save(best_model_state, best_model_save_path)

    # Save the training and validation loss logs as a JSON file
    log_file_path = os.path.join(session_dir, 'train_log.json')
    with open(log_file_path, 'w') as log_file:
        json.dump(loss_log, log_file)

    print('Training completed and logs saved')