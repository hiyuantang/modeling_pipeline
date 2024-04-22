from torch.utils.data import Dataset
from PIL import Image
import numpy as np
class PreprocessedDataset(Dataset):
    def __init__(self, preprocessed_data, transform=None):
        self.images, self.labels = zip(*preprocessed_data)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieve preprocessed images and labels
        image = self.images[idx][0].numpy()
        label = self.labels[idx][0].float()

        # Ensure the image has the correct shape (H, W, C)
        if image.ndim == 3 and image.shape[0] <= 50:  # Assuming (C, H, W)
            image = image.transpose((1, 2, 0))
        elif image.ndim == 3 and image.shape[-1] <= 50:  # Assuming (H, W, C)
            image = image.squeeze(-1)
        
        # Convert to 'uint8' data type if it's not already
        if image.dtype != np.uint8:
            # Normalize to the range [0, 255] if necessary
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        # Apply the specified transform
        if self.transform:
            image = self.transform(image)
        
        return image, label