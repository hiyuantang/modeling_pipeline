# FigureSynth Modeling Pipeline for Height Prediction

## Introduction
Welcome to the **FigureSynth Modeling Pipeline for Height Prediction**, a streamlined solution for training, testing, and visualizing deep learning models using synthetic children's data. For accessing the FigureSynth image dataset, visit the [Google Drive](https://drive.google.com/drive/folders/1G_iDkUxcRPQat-vAUkzzzOrmXC4p3-6H?usp=drive_link).

## Features
- **Data Pickle File Transformation**: Turn image data into pickle files for training, testing, and visualization purposes.
- **Automated Directory Management**: Automatically creates `results` and `vis` directories to store metadata, training logs, testing logs, model weights, and visualization diagrams, etc. Under the `results` directory, the script creates a session directory with a unique random hash for each training session.
- **Versatile Script**: Capable of performing training, testing, and visualization in a few lines of bash command.

## Getting Started
To get started with the pipeline, create a conda environment using the provided [`environment.yml`](environment.yml) file with the following command:

```bash
conda env create -f environment.yml
conda activate pytorch
```

# Usage

## Step 1: Prepare the Data
To download the image data, visit the [Google Drive](https://drive.google.com/drive/folders/1G_iDkUxcRPQat-vAUkzzzOrmXC4p3-6H?usp=drive_link).

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

## Step 4: Error Analysis

1. Open the Jupyter notebook titled [`error_analysis_demo.ipynb`](error_analysis_demo.ipynb)
2. Locate the line of code that reads:
```python
session_path = <your_modeling_session_path> # Define your session path
```
3. Replace `<your_modeling_session_path>` with the path to your modeling session
4. Run all cells in the notebook to execute the error analysis.


## Configuration Arguments

### For [`to_pickle.py`](src/to_pickle.py)
Customize the data processing with these arguments:

- **`--data_dir`**: Path to the dataset directory. Default: `E:/FigureSynth`.
- **`--data_name`**: Name of the dataset (e.g., `synth`, `kagglehw`). Default: `synth`.
- **`--image_type`**: Type of image (e.g., `depth`, `rgb`, `segmentation`). Default: `depth`.
- **`--gray_scale`**: Convert the image to grayscale by averaging over RGB channels. Default: `False`.
- **`--H_or_W`**: Create dataset label as height or weight (e.g., `H`, `W`). Default: `H`.
- **`--train_size`**: Proportion of the dataset for training. Default: `0.9`.

### For [`main.py`](src/main.py)
The `main.py` script offers a variety of command-line arguments for customization:

- **`--model_name`**: The model to use. Default: `resnet34`.
- **`--train_data_dir`**: Path of the training data. Default: `E:/synth_depth_False/trainset.pkl`.
- **`--test_data_dir`**: Path of the testing data. Default: `E:/synth_depth_False/testset.pkl`.
- **`--epochs`**: Number of training epochs. Default: `250`.
- **`--batch_size`**: Batch size for data processing. Default: `128`.
- **`--learning_rate`**: Learning rate for training. Default: `0.0001`.
- **`--drop_rate`**: Drop out rate. Default: `0.1`.
- **`--pre_trained_torchvision`**: Pre-trained weights for the model from torchvision. Default: `True`.
- **`--pre_trained_session_path`**: Pre-trained weights path for the model from session. Default: `None`.
- **`--update`**: Transfer Learning: Update all parameters or only the readout. Default: `all`.
- **`--device`**: Device for training (use `mps` for Mac). Default: `cuda`.
- **`--save_interval`**: Model save interval during training (set `-1` for no periodic save). Default: `-1`.
- **`--patience`**: Epochs to wait before early stopping. Default: `30`.
- **`--train_split`**: Dataset ratio for training. Default: `0.8`.
- **`--test_only`**: Perform only inference if `True`. Default: `False`.
- **`--session_path`**: Path to the trained model for testing (required if `--test_only` is `True`).

### For [`vis.py`](src/vis.py)
Configure the visualization of results with these command-line arguments:

- **`--all`**: If set to `True`, visualizes all sessions in the results directory. Default: `True`.
- **`--session_path`**: Specifies the path for visualizing a single session. This argument is not required if `--all` is set to `True`.

Remember to adjust these settings according to your project requirements and system configuration.

# Visualization Snapshot
![acc_plot](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/386142b2-9bd3-4f99-9317-a9cd3f6564e0)
![mae_plot](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/7e16b064-881b-4bcd-ab44-cae6f0844846)
![r2_plot](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/9ba7a287-467d-490e-8fd3-6371ac29f1ab)
![train_loss_plot](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/818ba8e7-7328-4866-9925-a7a0cdc07bc7)
![resnet50_gt_pred_plot](https://github.com/hiyuantang/modeling_pipeline/assets/24949723/48c580db-1665-41b6-8be2-dd0c3fcbe330)


# License
This project is licensed under the MIT License - see the [`LICENSE.md`](LICENSE) file for details.
