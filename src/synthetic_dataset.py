import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class SyntheticDataset(Dataset):
    """
    A dataset class for loading synthetic image data for machine learning models.

    Attributes:
        data_dir (str): Directory where the data files are stored.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        image_type (str): Type of image to process ('depth', 'rgb', or 'segmentation').
        gray_scale (bool): Flag to determine if the image should be convert the image to grayscale by averaging over RGB channels. 

    Methods:
        parse_file_name(file_name): Parses the file name to extract object ID and degree.
        __len__(): Returns the number of items in the dataset.
        __getitem__(idx): Retrieves an item from the dataset at the specified index.
    """

    def __init__(self, data_dir, transform=None, image_type='depth', gray_scale=False):
        """
        Initializes the SyntheticDataset with the given parameters.

        Args:
            data_dir (str): The directory where the data files are located.
            transform (callable, optional): Optional transform to be applied on a sample.
            image_type (str): The type of image to process ('depth', 'rgb', or 'segmentation').
            gray_scale (bool): If True, convert the image to grayscale by averaging over RGB channels. 
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_type = image_type
        self.gray_scale = gray_scale
        # Lists to store paths, degrees, and labels for each image type
        self.dpt_data = []
        self.rgb_data = []
        self.seg_data = []
        # Default dictionaries to store object-wise data
        self.dpt_data_obj = defaultdict(lambda: {'paths': [], 'degrees': [], 'label': None})
        self.rgb_data_obj = defaultdict(lambda: {'paths': [], 'degrees': [], 'label': None})
        self.seg_data_obj = defaultdict(lambda: {'paths': [], 'degrees': [], 'label': None})

        # Process all .txt files in the data directory to gather image paths and labels
        for root, _, files in os.walk(data_dir):
            for file_name in files:
                if file_name.endswith(".txt"):
                    object_id, degree = self.parse_file_name(file_name)
                    # Construct file names for each image type
                    dpt_file_name = f"child_{object_id:06d}_dpt_{degree:03d}.png"
                    rgb_file_name = f"child_{object_id:06d}_rgb_{degree:03d}.png"
                    seg_file_name = f"child_{object_id:06d}_seg_{degree:03d}.png"
                    # Construct full paths for each image
                    dpt_path = os.path.join(root, dpt_file_name)
                    rgb_path = os.path.join(root, rgb_file_name)
                    seg_path = os.path.join(root, seg_file_name)

                    # Read the label from the .txt file
                    with open(os.path.join(root, file_name), "r") as f:
                        first_line = f.readline()
                        # Extract the label using regex
                        label = float(re.findall(r"[-+]?\d*\.\d+|\d+", first_line.split(":")[1])[0])

                    # Update the lists with the new data
                    self.dpt_data.append((dpt_path, degree, label))
                    self.rgb_data.append((rgb_path, degree, label))
                    self.seg_data.append((seg_path, degree, label))

                    # Update the dictionaries with object-wise data
                    self.dpt_data_obj[object_id]['paths'].append(dpt_path)
                    self.dpt_data_obj[object_id]['degrees'].append(degree)
                    self.dpt_data_obj[object_id]['label'] = label

                    self.rgb_data_obj[object_id]['paths'].append(rgb_path)
                    self.rgb_data_obj[object_id]['degrees'].append(degree)
                    self.rgb_data_obj[object_id]['label'] = label

                    self.seg_data_obj[object_id]['paths'].append(seg_path)
                    self.seg_data_obj[object_id]['degrees'].append(degree)
                    self.seg_data_obj[object_id]['label'] = label

    def parse_file_name(self, file_name):
        """
        Parses the file name to extract the object ID and degree.

        Args:
            file_name (str): The name of the file to parse.

        Returns:
            tuple: A tuple containing the object ID and degree as integers.
        """
        # Regex pattern to match the file naming convention
        pattern = r"child_(\d+)_lbl_(\d+).txt"
        match = re.match(pattern, file_name)
        # Extract and return the object ID and degree
        object_id = int(match.group(1))
        degree = int(match.group(2))
        return object_id, degree

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: The number of items.
        """
        if self.image_type == 'depth':
            return len(self.dpt_data)
        elif self.image_type == 'rgb':
            return len(self.rgb_data)
        elif self.image_type == 'segmentation':
            return len(self.seg_data)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and label.
        """
        # Retrieve the item based on the image type
        if self.image_type == 'depth':
            return getitem_list_helper(self.dpt_data, idx, self.gray_scale, self.transform)
        elif self.image_type == 'rgb':
            return getitem_list_helper(self.rgb_data, idx, self.gray_scale, self.transform)
        elif self.image_type == 'segmentation':
            return getitem_list_helper(self.seg_data, idx, self.gray_scale, self.transform)
        else:
            raise ValueError("Invalid combination of image_type")

def getitem_list_helper(list_, idx, gray_scale, transform):
    """
    Helper function to get an item from a list-based dataset.

    Args:
        list_ (list): The dataset list containing paths, degrees, and labels.
        idx (int): The index of the item to retrieve.
        gray_scale (bool): If True, convert the image to grayscale by averaging over RGB channels.
        transform (callable): A function/transform that takes in an image and returns a transformed version.

    Returns:
        tuple: A tuple containing the image and label.
    """
    path, _, label = list_[idx]
    image = Image.open(path)

    if gray_scale:
        # Convert the image to grayscale by averaging over RGB channels
        image = image.convert("L")

    if transform:
        image = transform(image)
    
    label = torch.tensor(label, dtype=torch.float32)
    return image, label
