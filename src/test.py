import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from SyntheticDataset import SyntheticDataset
from tqdm import tqdm
import os
import json
from utils import *

def test(session_dir, test_data_dir, batch_size, device):

    model_path = os.path.join(session_dir, 'best_model.pth')
    info_path = os.path.join(session_dir, 'info.json')
    model_info = json2dict(info_path)
    
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5162, 0.5162, 0.5162], std=[0.2946, 0.2946, 0.2946]) 
    ])

    # Load your dataset
    test_dataset = SyntheticDataset(test_data_dir, transform=test_transform)

    # Create DataLoaders for each subset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the model
    model = get_pretrained_model(model_info['model_name'], num_classes=1, drop_rate=0.1, device=device, pretrained=False, print_summary=True)
    device = torch.device(device)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Initialize a dictionary to store the testing loss value
    loss_log = {'test_loss': []}

    # Initialize a dictionary to store the predictions and true labels
    predictions_log = {'true_labels': [], 'predictions': []}

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()

    # Inference
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Inference Progress:'):
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Store true labels and predictions
            predictions_log['true_labels'].extend(labels.tolist())
            predictions_log['predictions'].extend(outputs.tolist())

    # Calculate average validation loss
    test_loss /= len(test_loader)
    print(f'Inference Loss: {test_loss:.4f}')

    # Append the loss values for the current epoch to the loss_log dictionary
    loss_log['test_loss'].append(test_loss)

    # Save the loss_log dictionary as a JSON file
    log_file_path = os.path.join(session_dir, 'test_log.json')
    with open(log_file_path, 'w') as log_file:
        json.dump(loss_log, log_file)
    
    # Save the predictions_log dictionary as a JSON file
    pred_file_path = os.path.join(session_dir, 'pred.json')
    with open(pred_file_path, 'w') as pred_file:
        json.dump(predictions_log, pred_file)

    print('Inference completed and logs saved')