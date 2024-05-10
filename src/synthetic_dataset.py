import os
import re
from PIL import Image
import numpy as np
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
        mode (str, optional): Mode of operation, can be 'object' to handle object-wise data.
        expand (bool): Flag to determine if the image should be expanded to 3 channels.

    Methods:
        parse_file_name(file_name): Parses the file name to extract object ID and degree.
        __len__(): Returns the number of items in the dataset.
        __getitem__(idx): Retrieves an item from the dataset at the specified index.
    """

    def __init__(self, data_dir, transform=None, image_type='depth', mode=None, expand=True):
        """
        Initializes the SyntheticDataset with the given parameters.

        Args:
            data_dir (str): The directory where the data files are located.
            transform (callable, optional): Optional transform to be applied on a sample.
            image_type (str): The type of image to process ('depth', 'rgb', or 'segmentation').
            mode (str, optional): Mode of operation, can be 'object' to handle object-wise data.
            expand (bool): If True, expands the image to 3 channels.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_type = image_type
        self.mode = mode
        self.expand = expand
        # Lists to store paths, degrees, and labels for each image type
        self.dpt_data = []
        self.rgb_data = []
        self.seg_data = []
        # Default dictionaries to store object-wise data
        self.dpt_data_obj = defaultdict(lambda: {'paths': [], 'degrees': [], 'label': None})
        self.rgb_data_obj = defaultdict(lambda: {'paths': [], 'degrees': [], 'label': None})
        self.seg_data_obj = defaultdict(lambda: {'paths': [], 'degrees': [], 'label': None})

        # Process all .txt files in the data directory to gather image paths and labels
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".txt"):
                object_id, degree = self.parse_file_name(file_name)
                # Construct file names for each image type
                dpt_file_name = f"child_{object_id:06d}_dpt_{degree:03d}.png"
                rgb_file_name = f"child_{object_id:06d}_rgb_{degree:03d}.png"
                seg_file_name = f"child_{object_id:06d}_seg_{degree:03d}.png"
                # Construct full paths for each image
                dpt_path = os.path.join(data_dir, dpt_file_name)
                rgb_path = os.path.join(data_dir, rgb_file_name)
                seg_path = os.path.join(data_dir, seg_file_name)

                # Read the label from the .txt file
                with open(os.path.join(data_dir, file_name), "r") as f:
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
        Returns the number of items in the dataset based on the mode.

        Returns:
            int: The number of items.
        """
        # Return the length based on the selected mode
        if self.mode == None:
            return len(self.dpt_data)
        elif self.mode == 'object':
            return len(self.dpt_data_obj)
        else:
            pass  # Additional modes can be handled here

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and label.
        """
        # Retrieve the item based on the image type and mode
        if self.image_type == 'depth' and self.mode == None:
            return getitem_list_helper(self.dpt_data, idx, self.expand, self.transform)
        elif self.image_type == 'rgb' and self.mode == None:
            return getitem_list_helper(self.rgb_data, idx, self.expand, self.transform)
        elif self.image_type == 'segmentation' and self.mode == None:
            return getitem_list_helper(self.seg_data, idx, self.expand, self.transform)
        elif self.image_type == 'depthrgb' and self.mode != None:
            return getitem_dict_helper(self.dpt_data_obj, self.rgb_data_obj, idx, self.transform, self.mode)
        elif self.image_type == 'depthsegmentation' and self.mode != None:
            return getitem_dict_helper(self.dpt_data_obj, self.seg_data_obj, idx, self.transform, self.mode)
        elif self.image_type == 'rgbsegmentation' and self.mode != None:
            return getitem_dict_helper(self.rgb_data_obj, self.seg_data_obj, idx, self.transform, self.mode)
        else:
            raise ValueError("Invalid combination of image_type and mode")

# Helper functions are defined below with their respective docstrings and comments.

def getitem_list_helper(list_, idx, expand, transform):
    """
    Helper function to get an item from a list-based dataset.

    Args:
        list_ (list): The dataset list containing paths, degrees, and labels.
        idx (int): The index of the item to retrieve.
        expand (bool): If True, expands the image to 3 channels.
        transform (callable): A function/transform that takes in an image and returns a transformed version.

    Returns:
        tuple: A tuple containing the image and label.
    """
    path, _, label = list_[idx]
    image = Image.open(path)

    if not expand:
        # Convert the image to grayscale by averaging over RGB channels
        image = image.convert("L")

    if transform:
        image = transform(image)
    
    label = torch.tensor(label, dtype=torch.float32)
    return image, label

def getitem_dict_helper(dict0, dict1, idx, transform, mode):
    """
    Helper function to get an item from a dictionary-based dataset.

    Args:
        dict0 (defaultdict): The first dataset dictionary containing paths and labels.
        dict1 (defaultdict): The second dataset dictionary containing paths and labels.
        idx (int): The index of the item to retrieve.
        transform (callable): A function/transform that takes in an image and returns a transformed version.
        mode (str): The mode of operation, should be 'object' for object-wise data.

    Returns:
        tuple: A tuple containing the concatenated image arrays and label.
    """
    if mode == 'object':
        # Concatenate images from both dictionaries for the given object ID
        dict0_concat = []
        dict1_concat = []
        for path in dict0[idx]['paths']:
            dict0_img = Image.open(path)
            if transform:
                dict0_img = transform(dict0_img)
            dict0_concat.append(dict0_img)
          
        for path in dict1[idx]['paths']:
            dict1_img = Image.open(path)
            if transform:
                dict1_img = transform(dict1_img)
            dict1_concat.append(dict1_img)
    
        # Concatenate the image arrays along the last axis
        dict0_array = np.concatenate(dict0_concat, axis=-1)
        dict1_array = np.concatenate(dict1_concat, axis=-1)
        
    label = torch.tensor(dict0[idx]['label'], dtype=torch.float32)
    concat_array = np.concatenate((dict0_array, dict1_array), axis=-1)

    return concat_array, label
