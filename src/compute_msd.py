import argparse
import torch
from torchvision import transforms
from tqdm import tqdm
from utils import create_dataloader_from_preprocessed

def main():
    parser = argparse.ArgumentParser(description='Process dataset arguments')
    parser.add_argument('--data_dir', type=str, default="E:/depth_None_True/testset.pkl", help='Path to the dataset directory')
    args = parser.parse_args()

    # Define a transform to convert images to tensors
    to_tensor = transforms.Compose([transforms.ToTensor()])
    
    print('...loading full dataset')
    loader = create_dataloader_from_preprocessed(args.data_dir, 1, transform = to_tensor)
    print('...data successfully loaded from the pickle file.')

    mean = torch.zeros(3)
    std = torch.zeros(3)

    for data, _ in tqdm(loader, desc='Progress'):
        mean += data.mean([0, 2, 3])
        std += data.std([0, 2, 3])

    mean /= len(loader)
    std /= len(loader)

    # Print the computed mean and standard deviation
    print(f'Mean: {mean}')
    print(f'Std: {std}')

if __name__=='__main__':
    main()