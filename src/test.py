import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import json
from utils import *

def test(session_dir, test_data_dir, batch_size, drop_rate, device):
    """
    Perform inference on a test dataset using a pre-trained model.

    This function loads a trained model and performs inference on a provided
    test dataset. It computes the loss using Mean Squared Error (MSE) and
    saves the inference results and loss values in JSON files.

    Parameters:
        session_dir (str): Directory where the session's files are stored.
        test_data_dir (str): Directory containing the preprocessed test data.
        batch_size (int): The size of the batch for the DataLoader.
        drop_rate (float): The dropout rate used in the model.
        device (str): The device to run the inference on ('cpu' or 'cuda').
    """
    # Paths to the model and info files
    model_path = os.path.join(session_dir, 'best_model.pth')
    info_path = os.path.join(session_dir, 'info.json')
    # Load model information from the JSON file
    model_info = json2dict(info_path)
    
    # Define the transformations for the test dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((224, 224)),  
        # transforms.Normalize(mean=[0.3568, 0.3568, 0.3568], std=[0.3512, 0.3512, 0.3512]), # Means and Standard Deviations for depth maps
        transforms.Normalize(mean=[0.2341, 0.2244, 0.2061], std=[0.1645, 0.1472, 0.1261]), # Means and Standard Deviations for RGB images
    ])

    # Load the test dataset
    print('...loading testing dataset')
    test_dataset = create_dataset_from_preprocessed(test_data_dir, test_transform)

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    print('...testing dataset loading completed')

    # Load the model for inference
    model, _ = get_pretrained_model(model_info['model_name'], num_classes=1, drop_rate=drop_rate, batch_size=batch_size, pretrained=False)
    device = torch.device(device)
    model = model.to(device)
    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Initialize logs for storing loss and predictions
    loss_log = {'test_loss': []}
    predictions_log = {'true_labels': [], 'predictions': []}

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Perform inference on the test dataset
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Inference Progress:'):
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
            # Get model predictions
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Store true labels and model predictions
            predictions_log['true_labels'].extend(labels.tolist())
            predictions_log['predictions'].extend(outputs.tolist())

    # Calculate the average loss over all batches
    test_loss /= len(test_loader)
    print(f'Inference Loss: {test_loss:.4f}')

    # Save the loss and predictions to JSON files
    loss_log['test_loss'].append(test_loss)
    log_file_path = os.path.join(session_dir, 'test_log.json')
    with open(log_file_path, 'w') as log_file:
        json.dump(loss_log, log_file)
    
    pred_file_path = os.path.join(session_dir, 'pred.json')
    with open(pred_file_path, 'w') as pred_file:
        json.dump(predictions_log, pred_file)

    print('Inference completed and logs saved')
