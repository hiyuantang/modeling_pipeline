import os
os.makedirs('./data', exist_ok=True)
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pickle
from SyntheticDataset import SyntheticDataset
from pickle_dataset import PreprocessedDataset
from utils import *
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process dataset arguments')
    parser.add_argument('--data_dir', type=str, default='E:\synthetic_kids', help='Path to the dataset directory')
    parser.add_argument('--image_type', type=str, default='depth', help='Type of image (e.g., depth, color)')
    parser.add_argument('--mode', default=None, help='Dataset mode')
    parser.add_argument('--expand', type=bool, default=False, help='Expand the dataset')
    args = parser.parse_args()

    # Image Dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor()
        ])

    data_set = SyntheticDataset(args.data_dir, transform=transform, image_type=args.image_type, mode=args.mode, expand=args.expand)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, pin_memory=True)

    # Custom directory name based on the provided arguments
    custom_dir_name = f"{args.image_type}_{args.mode}_{args.expand}"

    # Create the custom directory if it doesn't exist
    custom_dir_path = os.path.join(os.path.dirname(args.data_dir), custom_dir_name)
    os.makedirs(custom_dir_path, exist_ok=True)

    # Preprocess and save the dataset
    pkl_path = os.path.join(custom_dir_path, 'data_loader.pkl')
    preprocess_and_save_dataset(data_loader, pkl_path)

    # Pickle Dataloader
    # batch_size = 256
    # train_dataloader = create_dataloader_from_preprocessed(pkl_path, batch_size)
    # print('Data successfully loaded from the pickle file.')

if __name__=='__main__':
    main()