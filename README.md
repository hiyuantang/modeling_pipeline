# Synthetic Kids' Data Modeling Pipeline for Height Prediction

## Introduction
Welcome to the **Synthetic Kids' Data Modeling Pipeline for Height Prediction**, a streamlined solution for training, testing, and visualizing deep learning models using synthetic children's data. For more details, visit the project repository here.

## Features
- **Data Pickle File Transformation**: Turn image data into pickle files for training, testing, and visualization purposes.
- **Automated Directory Management**: Automatically creates `results` and `vis` directories to store metadata, training logs, testing logs, model weights, and visualization diagrams, etc. Under the `results` directory, the script creates a session directory with a unique random hash for each training session.
- **Versatile Script**: Capable of performing training, testing, and visualization in a few lines of bash command.

## Getting Started
To get started with the pipeline, create a conda environment using the provided `environment.yml` file with the following command:

```bash
conda env create -f environment.yml
```

# Usage

## Step 1: Prepare the Data
To download the image data, visit the project repository.

**Transform the image dataset into a pickle file for faster data loading:**
```bash
python3 src/to_pickle.py --data_dir <your_image_data_path> --train_size 0.9
```

## Step 2: Train the Model

**Train the model, save the best model according to the validation set, and make inferences on test data:**
```bash
python3 src/main.py --model_name vgg16 --train_data_dir <your_data_path> --epochs 200 --batch_size 64 --device cuda
```

## Step 3: Visualize the Results

**Visualize all modeling sessions at once:**
```bash
python3 src/vis.py --all True
```

**Visualize a single modeling session:**
```bash
python3 src/vis.py --all False --session_path <your_modeling_session_path>
```

Please replace `<your_image_data_path>` and `<your_data_path>` with the actual paths to your data, and `<your_modeling_session_path>` with the path to your specific modeling session.

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

# Visualization Snapshot
![output](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/1d37f106-ab0c-4ba9-9012-eb315e5e74f2)
![output](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/d700bac9-059a-40ee-8b46-b82e0a35606c)
![output](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/528ad12e-1dd4-4b29-b9a9-0e06e12e680b)

# License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
