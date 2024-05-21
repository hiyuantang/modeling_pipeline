import argparse
import torch
from torchvision import transforms
from tqdm import tqdm
from utils import create_dataloader_from_preprocessed

def main():
    """
    Compute the mean and standard deviation of a dataset.

    This script calculates the mean and standard deviation of the RGB channels
    in the dataset. It assumes that the dataset is stored in a pickle file and
    uses a DataLoader to iterate through the dataset.
    """
    # Set up argument parser for command line arguments
    parser = argparse.ArgumentParser(description='Process dataset arguments')
    # Add data directory argument with a default path
    parser.add_argument('--data_dir', type=str, default='E:/synth_rgb_False/trainset.pkl', help='Path to the dataset directory')
    args = parser.parse_args()

    # Define a transform to convert images to tensors
    to_tensor = transforms.Compose([transforms.ToTensor()])
    
    # Load the dataset from the specified directory
    print('...loading full dataset')
    loader = create_dataloader_from_preprocessed(args.data_dir, 1, transform=to_tensor)
    print('...data successfully loaded from the pickle file.')

    # Initialize tensors to hold the mean and standard deviation
    mean = torch.zeros(3)
    std = torch.zeros(3)

    # Iterate over the dataset to compute the mean and standard deviation
    for data, _ in tqdm(loader, desc='Progress'):
        # Calculate the mean and standard deviation for each batch and accumulate
        mean += data.mean([0, 2, 3])
        std += data.std([0, 2, 3])

    # Average the mean and standard deviation over all batches
    mean /= len(loader)
    std /= len(loader)

    # Print the computed mean and standard deviation for the dataset
    print(f'Mean: {mean}')
    print(f'Std: {std}')

# Entry point of the script
if __name__=='__main__':
    main()
