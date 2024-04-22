from torch.utils.data import Dataset

class PreprocessedDataset(Dataset):
    def __init__(self, preprocessed_data, transform=None):
        self.images, self.labels = zip(*preprocessed_data)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieve preprocessed images and labels
        image = self.images[idx][0].float()
        label = self.labels[idx][0].float()
        
        # Apply the specified transform
        if self.transform:
            image = self.transform(image)
        
        return image, label