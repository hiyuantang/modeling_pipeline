import os
import re
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class SyntheticDataset(Dataset):
    def __init__(self, data_dir, transform=None, image_type='depth', mode=None, expand=True):
        self.data_dir = data_dir
        self.transform = transform
        self.image_type = image_type
        self.mode = mode
        self.expand = expand
        self.dpt_data = []
        self.rgb_data = []
        self.seg_data = []
        self.dpt_data_obj = defaultdict(lambda: {'paths': [], 'degrees': [], 'label': None})
        self.rgb_data_obj = defaultdict(lambda: {'paths': [], 'degrees': [], 'label': None})
        self.seg_data_obj = defaultdict(lambda: {'paths': [], 'degrees': [], 'label': None})

        # Iterate through all .txt files in the data directory
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".txt"):
                object_id, degree = self.parse_file_name(file_name)
                dpt_file_name = f"child_{object_id:06d}_dpt_{degree:03d}.png"
                rgb_file_name = f"child_{object_id:06d}_rgb_{degree:03d}.png"
                seg_file_name = f"child_{object_id:06d}_seg_{degree:03d}.png"
                dpt_path = os.path.join(data_dir, dpt_file_name)
                rgb_path = os.path.join(data_dir, rgb_file_name)
                seg_path = os.path.join(data_dir, seg_file_name)

                # Read the height label from the .txt file
                with open(os.path.join(data_dir, file_name), "r") as f:
                    first_line = f.readline()
                    label = float(re.findall(r"[-+]?\d*\.\d+|\d+", first_line.split(":")[1])[0])

                # Update data
                self.dpt_data.append((dpt_path, degree, label))
                self.rgb_data.append((rgb_path, degree, label))
                self.seg_data.append((seg_path, degree, label))

                # Update data by object
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
        # Extract object ID and degree from the file name
        pattern = r"child_(\d+)_lbl_(\d+).txt"
        match = re.match(pattern, file_name)
        object_id = int(match.group(1))
        degree = int(match.group(2))
        return object_id, degree

    def __len__(self):
        if self.mode == None:
            return len(self.dpt_data)
        elif self.mode == 'object':
            return len(self.dpt_data_obj)
        else:
            pass

    def __getitem__(self, idx):
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
    

def getitem_list_helper(list_, idx, expand, transform):
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
    if mode == 'object':
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
    
        dict0_array = np.concatenate(dict0_concat, axis=-1)
        dict1_array = np.concatenate(dict1_concat, axis=-1)
        
    label = torch.tensor(dict0[idx]['label'], dtype=torch.float32)
    concat_array = np.concatenate((dict0_array, dict1_array), axis=-1)

    return concat_array, label
