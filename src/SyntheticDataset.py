import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []

        # Iterate through all .txt files in the data directory
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".txt"):
                object_id, degree = self.parse_file_name(file_name)
                img_file_name = f"depth{object_id:06d}_{degree:03d}.png"
                img_path = os.path.join(data_dir, img_file_name)

                # Read the label from the .txt file
                with open(os.path.join(data_dir, file_name), "r") as f:
                    label = float(re.findall(r"[-+]?\d*\.\d+|\d+", f.read().split(":")[1])[0])

                self.data.append((img_path, label))

    def parse_file_name(self, file_name):
        # Extract object ID and degree from the file name
        pattern = r"scene(\d+)_(\d+).txt"
        match = re.match(pattern, file_name)
        object_id = int(match.group(1))
        degree = int(match.group(2))
        return object_id, degree

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Convert the label to float32
        label = torch.tensor(label, dtype=torch.float32)

        return img, label