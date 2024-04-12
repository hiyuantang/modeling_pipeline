import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from SyntheticDataset import SyntheticDataset

# Define a transform to convert images to tensors
to_tensor = transforms.Compose([transforms.ToTensor()])
train_data_dir = 'E:/db_synthetic_1'
dataset = SyntheticDataset(train_data_dir, transform=to_tensor)
loader = DataLoader(dataset, batch_size=1, num_workers=0)

mean = torch.zeros(3)
std = torch.zeros(3)

for data, _ in loader:
    mean += data.mean([0, 2, 3])
    std += data.std([0, 2, 3])

mean /= len(loader)
std /= len(loader)

# Print the computed mean and standard deviation
print(f'Mean: {mean}')
print(f'Std: {std}')