import os
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from synthetic_dataset import SyntheticDataset
from kaggle_hw_dataset import Kaggle_HW_Dataset
from utils import *
import argparse

def main():
    """
    Prepare and preprocess a synthetic dataset for training and testing.

    This script takes a directory containing synthetic images, applies transformations,
    splits the dataset into training and testing sets, and saves them as pickle files
    for later use in training and testing machine learning models.

    The script is configurable via command line arguments for flexibility in specifying
    the dataset directory, image type, channel expansion, and train-test split ratio.
    """
    # Set up argument parser for command line arguments
    parser = argparse.ArgumentParser(description='Process dataset arguments')
    # Add arguments for dataset configuration
    parser.add_argument('--data_dir', type=str, default='E:/kaggle_height_weight', help='Path to the dataset directory')
    parser.add_argument('--data_name', type=str, choices=['synth', 'kagglehw'], default='kagglehw', help='Name of the dataset')
    parser.add_argument('--image_type', type=str, default='rgb', help='Type of image (e.g., depth, rgb, or segmentation)')
    parser.add_argument('--gray_scale', type=bool, default=False, help='Convert the image to grayscale by averaging over RGB channels')
    parser.add_argument('--H_or_W', type=str, default='H', choices=['H', 'W'], help='Create dataset label as height or weight')
    parser.add_argument('--train_size', type=float, default=0.9, help='Proportion of dataset to use for training')
    args = parser.parse_args()

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
    ])
    
    # Load the full synthetic dataset
    print('...loading full dataset')
    if args.data_name == 'synth':
        full_dataset = SyntheticDataset(args.data_dir, transform=transform, image_type=args.image_type, gray_scale=args.gray_scale)
    elif args.data_name == 'kagglehw':
        full_dataset = Kaggle_HW_Dataset(args.data_dir, transform=transform, gray_scale=args.gray_scale, label=args.H_or_W)
    print('...full dataset loading completed')
    
    # Split the dataset into training and testing subsets
    train_size = int(len(full_dataset) * args.train_size)
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create DataLoaders for the training and testing sets
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Generate a custom directory name based on the provided arguments
    custom_dir_name = f"{args.data_name}_{args.image_type}_{str(args.gray_scale)}"
    custom_dir_path = os.path.join(os.path.dirname(args.data_dir), custom_dir_name)
    # Ensure the directory exists
    os.makedirs(custom_dir_path, exist_ok=True)

    # Preprocess and save the training dataset as a pickle file
    train_pkl_path = os.path.join(custom_dir_path, 'trainset.pkl')
    preprocess_and_save_dataset(train_loader, train_pkl_path)

    # Preprocess and save the testing dataset as a pickle file
    test_pkl_path = os.path.join(custom_dir_path, 'testset.pkl')
    preprocess_and_save_dataset(test_loader, test_pkl_path)

# Entry point of the script
if __name__=='__main__':
    main()
