#!/bin/bash

python src/main.py --model_name baseline_cnn
python src/main.py --model_name resnet18
python src/main.py --model_name resnet34
python src/main.py --model_name resnet50
python src/main.py --model_name mobilenet_v3_small
python src/main.py --model_name mobilenet_v3_large