# Modeling Pipeline

## Introduction
Welcome to the `Modeling Pipeline`, a comprehensive solution for training, testing, and visualizing machine learning models. The main script, `main.py`, streamlines the process of managing your machine learning experiments.

## Features
- **Automated Directory Management**: Automatically creates a `results` directory and a session directory with a unique random hash for each training session.
- **Versatile Script**: Capable of performing training, testing, and visualization in a single run or individually as needed.

## Getting Started
To get started with `main.py`, navigate to the `src` directory of the project:

```bash
cd path/to/src
```

# Prerequisites
Ensure you have the following prerequisites installed:
- Python 3.11
- matplotlib
- numpy
- scipy
- scikit-learn
- torch
- torchvision
- torchsummary
- tqdm
- pillow

# Usage
Run `main.py` using the following command:
```bash
python3 main.py --model_name vgg16 --train_data_dir your_data_path --epochs 200 --batch_size 64 --device cuda 
```

# Customization
`main.py` uses `argparse` for customization. The current version supports the following arguments:

## Configuration Arguments
The `main.py` script is highly customizable through the use of command-line arguments. Below is a list of available arguments along with their descriptions and default values:
- **`--model_name`**: Specifies the name of the model to be used. Default is `resnet34`.
- **`--train_data_dir`**: Sets the directory where the training data is located. Default is `E:\\db_synthetic_1`.
- **`--epochs`**: Determines the number of epochs for training the model. Default is `100`.
- **`--batch_size`**: Defines the size of each batch of data to be processed. Default is `128`.
- **`--learning_rate`**: Sets the learning rate for the training process. Default is `0.0001`.
- **`--device`**: Chooses the device for training the model. Default is `cuda`. Mac users should replace this with `mps` to utilize Metal Performance Shaders.
- **`--save_interval`**: The interval at which the model is saved during training. Default is `-1`, which means the model is not saved periodically.
- **`--patience`**: The number of epochs to wait for improvement before early stopping. Default is `30`.
- **`--train_split`**: The ratio of the dataset to be used for training. Default is `0.8`, meaning 80% for training and 20% for validation.
- **`--test_only`**: When set to `True`, the script will only perform model inference without training. Default is `False`.
- **`--session_path`**: Specifies the path to the model that should be tested. This argument is required when `--test_only` is set to `True`, to indicate which trained model to use for testing.

Remember to adjust these settings according to your project requirements and system configuration.

# Visualization
![output](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/1d37f106-ab0c-4ba9-9012-eb315e5e74f2)
![output](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/d700bac9-059a-40ee-8b46-b82e0a35606c)
![output](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/528ad12e-1dd4-4b29-b9a9-0e06e12e680b)
![output](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/41f5ce07-f205-43a8-8e7b-7b2a1d4c8103)

# License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
