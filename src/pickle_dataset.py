import os
import pickle
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PickleDataset(Dataset):
    """
    A custom Dataset class for loading images and labels from pickle files.

    Attributes:
        pickle_dir (str): Directory containing the pickle files.
        transform (callable, optional): Optional transform to be applied on a sample.
        pickle_files (list): List of paths to the pickle files.
        images (list): List of all images loaded from the pickle files.
        labels (list): List of all labels loaded from the pickle files.
    """

    def __init__(self, pickle_dir, transform=None):
        """
        Args:
            pickle_dir (str): Directory with all the pickle files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.pickle_dir = pickle_dir
        self.transform = transform
        self.pickle_files = [os.path.join(pickle_dir, f) for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
        self.images, self.labels = self._load_all_images()

    def _load_all_images(self):
        """
        Load all images and labels from the pickle files.

        Returns:
            tuple: A tuple containing two lists: images and labels.
        """
        images = []
        labels = []
        for file in self.pickle_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                images.extend(data[0])
                labels.extend(data[1])
        return images, labels

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to be fetched.

        Returns:
            tuple: (image, label) where image is a PIL image and label is the corresponding label.
        """
        tensor_image = self.images[idx]
        numpy_image = tensor_image.permute(1, 2, 0).numpy()  # Convert tensor to numpy array
        numpy_image = (numpy_image * 255).astype(np.uint8)  # Convert to uint8 type
        image = Image.fromarray(numpy_image)  # Create PIL Image
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label

