# Synthetic Kids' Data Modeling Pipeline for Height Prediction

## Introduction
Welcome to the **Synthetic Kids' Data Modeling Pipeline for Height Prediction**, a streamlined solution for training, testing, and visualizing deep learning models using synthetic children's data. For accessing the synthetic kids' image dataset, visit the [repository](https://github.com/davidberth/ac297r_project6).

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
To download the image data, visit the [repository](https://github.com/davidberth/ac297r_project6).

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

## Configuration Arguments

### For `to_pickle.py`
Customize the data processing with these arguments:

- **`--data_dir`**: Path to the dataset directory. Default: `E:\\synthetic_kids`.
- **`--image_type`**: Type of image (e.g., `depth`, `color`). Default: `depth`.
- **`--mode`**: Dataset mode. Default: `None`.
- **`--expand`**: Maintain 3 channels or average to 1 channel. Default: `True`.
- **`--train_size`**: Proportion of the dataset for training. Default: `0.9`.

### For `main.py`
The `main.py` script offers a variety of command-line arguments for customization:

- **`--model_name`**: The model to use. Default: `resnet34`.
- **`--train_data_dir`**: The training data directory. Default: `E:\\db_synthetic_1`.
- **`--epochs`**: Number of training epochs. Default: `100`.
- **`--batch_size`**: Batch size for data processing. Default: `128`.
- **`--learning_rate`**: Learning rate for training. Default: `0.0001`.
- **`--device`**: Device for training (use `mps` for Mac). Default: `cuda`.
- **`--save_interval`**: Model save interval during training (set `-1` for no periodic save). Default: `-1`.
- **`--patience`**: Epochs to wait before early stopping. Default: `30`.
- **`--train_split`**: Dataset ratio for training. Default: `0.8`.
- **`--test_only`**: Perform only inference if `True`. Default: `False`.
- **`--session_path`**: Path to the trained model for testing (required if `--test_only` is `True`).

### For `vis.py`
Configure the visualization of results with these command-line arguments:

- **`--all`**: If set to `True`, visualizes all sessions in the results directory. Default: `True`.
- **`--session_path`**: Specifies the path for visualizing a single session. This argument is not required if `--all` is set to `True`.

Remember to adjust these settings according to your project requirements and system configuration.

# Visualization Snapshot
![output](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/1d37f106-ab0c-4ba9-9012-eb315e5e74f2)
![output](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/d700bac9-059a-40ee-8b46-b82e0a35606c)
![output](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/528ad12e-1dd4-4b29-b9a9-0e06e12e680b)

# License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
