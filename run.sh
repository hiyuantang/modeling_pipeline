#!/bin/bash

python src/main.py --model_name baseline_cnn --epochs 100
python src/main.py --model_name resnet18 --epochs 100
python src/main.py --model_name resnet34 --epochs 100
python src/main.py --model_name resnet50 --epochs 100
python src/main.py --model_name resnet101 --epochs 100
python src/main.py --model_name resnet152 --epochs 100
python src/main.py --model_name vgg16 --epochs 100
python src/main.py --model_name vgg19 --epochs 100
python src/main.py --model_name vgg16_bn --epochs 100
python src/main.py --model_name vgg19_bn --epochs 100
python src/main.py --model_name mobilenet_v3_small --epochs 100
python src/main.py --model_name mobilenet_v3_large --epochs 100
python src/main.py --model_name vit_b_16 --epochs 100
python src/main.py --model_name vit_b_32 --epochs 100
python src/main.py --model_name vit_l_16 --epochs 100
python src/main.py --model_name vit_l_32 --epochs 100
python src/main.py --model_name vit_h_14 --epochs 100