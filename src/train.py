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

def train(model_name, train_data_dir, epochs, batch_size, learning_rate, device, 
          save_interval, patience, train_split, session_dir):
    # Load the dataset
    train_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(15), 
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  
    transforms.Normalize(mean=[0.3568, 0.3568, 0.3568], std=[0.3512, 0.3512, 0.3512]) 
    ])

    val_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),   
    transforms.Normalize(mean=[0.3568, 0.3568, 0.3568], std=[0.3512, 0.3512, 0.3512]) 
    ])

    # Load your dataset
    print('...training full dataset')
    dataset = create_dataset_from_preprocessed(train_data_dir, None)

    # Split dataset into training and validation sets
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply the appropriate transform to each subset
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create DataLoaders for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print('...training dataset loading completed')

    # Early stopping initialization
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # Initialize a dictionary to store the loss values for each epoch
    loss_log = {'train_loss': [], 'val_loss': []}

    # Load the model
    model = get_pretrained_model(model_name, num_classes=1, drop_rate=0.1, device=device, pretrained=True, print_summary=True)
    device = torch.device(device)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
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

            # Save the model at the specified interval
            if save_interval == -1:
                continue
            else:
                if (i + 1) % save_interval == 0:
                    model_save_path = os.path.join(session_dir, f'{model_name}_step_{i+1}.pth')
                    torch.save(model.state_dict(), model_save_path)

        # Print statistics
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs} - Training Loss: {epoch_loss:.4f}')

        # Validation loop
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Calculate average validation loss
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')

        # Append the loss values for the current epoch to the loss_log dictionary
        loss_log['train_loss'].append(epoch_loss)
        loss_log['val_loss'].append(val_loss)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Update the best model state
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {patience} epochs without improvement.')
                break
    if best_model_state is not None:
        best_model_save_path = os.path.join(session_dir, 'best_model.pth')
        torch.save(best_model_state, best_model_save_path)

    # Save the loss_log dictionary as a JSON file
    log_file_path = os.path.join(session_dir, 'loss_log.json')
    with open(log_file_path, 'w') as log_file:
        json.dump(loss_log, log_file)

    print('Training completed and logs saved')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    train('vgg16', 'E:\db_synthetic_1', epochs=10, batch_size=32, learning_rate=0.001, device=device, save_interval=-1, patience=15, train_split=0.8)
