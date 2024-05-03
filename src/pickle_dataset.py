from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PreprocessedDataset(Dataset):
    """
    A dataset class for handling preprocessed image data.

    This class inherits from PyTorch's Dataset and is used to interface with
    preprocessed image data for training or testing in a machine learning model.

    Attributes:
        images (list): A list of preprocessed images.
        labels (list): A list of labels corresponding to the images.
        transform (callable, optional): An optional transform to be applied
            on a sample.

    Parameters:
        preprocessed_data (list): A list of tuples where each tuple contains
            an image and its label.
        transform (callable, optional): An optional transform to be applied
            on a sample.
    """
    def __init__(self, preprocessed_data, transform=None):
        # Initialize dataset with images and labels
        self.images, self.labels = zip(*preprocessed_data)
        self.transform = transform

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a sample and its label from the dataset.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and its label.
        """
        # Retrieve preprocessed images and labels
        image = self.images[idx][0].numpy()
        label = self.labels[idx][0].float()

        # Ensure the image has the correct shape (H, W, C)
        if image.ndim == 3 and image.shape[0] <= 50:  # Assuming (C, H, W)
            # Convert from (C, H, W) to (H, W, C)
            image = image.transpose((1, 2, 0))
        elif image.ndim == 3 and image.shape[-1] <= 50:  # Assuming (H, W, C)
            # Remove the last dimension if it's a single-channel image
            image = image.squeeze(-1)
        
        # Convert to 'uint8' data type if it's not already
        if image.dtype != np.uint8:
            # Normalize to the range [0, 255] if necessary
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)
        
        # Convert to PIL Image for further processing
        image = Image.fromarray(image)
        
        # Apply the specified transform, if any
        if self.transform:
            image = self.transform(image)
        
        return image, label
