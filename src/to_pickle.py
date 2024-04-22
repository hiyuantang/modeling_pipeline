import os
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from synthetic_dataset import SyntheticDataset
from utils import *
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process dataset arguments')
    parser.add_argument('--data_dir', type=str, default='E:\synthetic_kids', help='Path to the dataset directory')
    parser.add_argument('--image_type', type=str, default='depth', help='Type of image (e.g., depth, color)')
    parser.add_argument('--mode', default=None, help='Dataset mode')
    parser.add_argument('--expand', type=bool, default=True, help='Keep image as having 3 channels or average to a single channel')
    parser.add_argument('--train_size', type=float, default=0.9, help='Proportion of dataset to use for training')
    args = parser.parse_args()

    # Image Dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor()
        ])
    
    # Load the full dataset
    print('...loading full dataset')
    full_dataset  = SyntheticDataset(args.data_dir, transform=transform, image_type=args.image_type, mode=args.mode, expand=args.expand)
    print('...full dataset loading completed')
    
    # Split the dataset into training and testing sets
    train_size = int(len(full_dataset) * args.train_size)
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # DataLoaders for training and testing sets
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Custom directory name based on the provided arguments
    custom_dir_name = f"{args.image_type}_{args.mode}_{str(args.expand)}"
    custom_dir_path = os.path.join(os.path.dirname(args.data_dir), custom_dir_name)
    os.makedirs(custom_dir_path, exist_ok=True)

    # Preprocess and save the training dataset
    train_pkl_path = os.path.join(custom_dir_path, 'trainset.pkl')
    preprocess_and_save_dataset(train_loader, train_pkl_path)

    # Preprocess and save the testing dataset
    test_pkl_path = os.path.join(custom_dir_path, 'testset.pkl')
    preprocess_and_save_dataset(test_loader, test_pkl_path)

if __name__=='__main__':
    main()