# Modeling Pipeline

## Introduction
Welcome to the Project Name, a comprehensive solution for training, testing, and visualizing machine learning models. The main script, `main.py`, streamlines the process of managing your machine learning experiments.

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

# Usage
Run `main.py` using the following command:
`python3 main.py --model_name vgg16`

# Customization
`main.py` uses `argparse` for customization. The current version supports the following arguments:

Note: The default device is set to `'cuda'`. If you are using a Mac, replace `'cuda'` with `'mps'` to utilize Apple’s Metal Performance Shaders for GPU acceleration.
 
# License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
