#!/bin/bash

python src/main.py --model_name baseline_cnn
python src/main.py --model_name resnet18
python src/main.py --model_name resnet34
python src/main.py --model_name resnet50
python src/main.py --model_name resnet101
python src/main.py --model_name resnet152
python src/main.py --model_name vgg16
python src/main.py --model_name vgg19
python src/main.py --model_name vgg16_bn
python src/main.py --model_name vgg19_bn
python src/main.py --model_name mobilenet_v3_small
python src/main.py --model_name mobilenet_v3_large
python src/main.py --model_name vit_b_16
python src/main.py --model_name vit_b_32
python src/main.py --model_name vit_l_16
python src/main.py --model_name vit_l_32
python src/main.py --model_name vit_h_14