import os
import re
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import pandas as pd

class Kaggle_HW_Dataset(Dataset):
    def __init__(self, data_dir, transform=None, gray_scale=False, label='H'):
        self.data_dir = data_dir
        self.transform = transform
        self.gray_scale = gray_scale
        self.label_option = label
        if label=='H':
            print('Generating dataset with Height labels.')
        elif label=='W':
            print('Generating dataset with Weight labels.')
        else:
            raise ValueError("Label must be 'H' for height or 'W' for weight")

        df_meta_data = pd.read_csv(os.path.join(data_dir, 'Output_data.csv'))
        list_labels = df_meta_data['Height & Weight'].to_list()

        self.data_paths = df_meta_data['Filename'].to_list()
        self.labels_h = []
        self.labels_w = []

        for item in list_labels:
            height, weight = self._parse_height_weight(item)
            self.labels_h.append(height)
            self.labels_w.append(weight)
    
    def _parse_height_weight(self, hw_str):
        # Example format: "4' 10\" 170 lbs."
        height_match = re.match(r" (\d+)' (\d+)\"", hw_str)
        weight_match = re.search(r"(\d+) lbs.", hw_str)
        
        if height_match and weight_match:
            feet = int(height_match.group(1))
            inches = int(height_match.group(2))
            height_in_inches = feet * 12 + inches  # Convert height to inches
            height_in_cm = height_in_inches * 2.54  # Convert height to centimeters
            weight = int(weight_match.group(1))
            return height_in_cm, weight
        else:
            raise ValueError(f"Invalid height and weight format: {hw_str}")
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_paths[idx])
        image = Image.open(data_path)

        if self.gray_scale:
            image = image.convert("L")
        
        # Zero-pad the image to make it square
        max_side = max(image.size)
        padding = (
            (max_side - image.size[0]) // 2,  # left
            (max_side - image.size[1]) // 2,  # top
            (max_side - image.size[0] + 1) // 2,  # right
            (max_side - image.size[1] + 1) // 2   # bottom
        )
        image = ImageOps.expand(image, padding, fill=0)

        # Resize the image to 224x224
        image = image.resize((224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        if self.label_option == 'H':
            label = torch.tensor(self.labels_h[idx], dtype=torch.float32)
        else:
            label = torch.tensor(self.labels_w[idx], dtype=torch.float32)
        return image, label