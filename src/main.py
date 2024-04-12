import os
os.makedirs('./results', exist_ok=True)
import argparse
import json
from train import train
from test import test
from utils import *

def main():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--model_name', type=str, default='resnet34', help='Name of the model')
    parser.add_argument('--train_data_dir', type=str, default='E:\db_synthetic_1', help='Directory of the training data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    parser.add_argument('--save_interval', type=int, default=-1, help='Interval to save the model')
    parser.add_argument('--patience', type=int, default=30, help='Patience for early stopping')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--test_only', type=bool, default=False, help='Make model inference')
    parser.add_argument('--session_path', type=str, required=False, help='Tell which model to test')
    args = parser.parse_args()

    if args.test_only==False:

        # Generate a unique hash for the current training session
        session_results_dir = generate_unique_hash()
        os.makedirs(session_results_dir, exist_ok=True)

        # Create a dictionary with model information
        model_info = {
            'id': session_results_dir, 
            'model_name': args.model_name,
            'train_data_dir': args.train_data_dir,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'device': args.device,
            'save_interval': args.save_interval,
            'patience': args.patience,
            'train_split': args.train_split
        }

        # Save the model information to info.json within the session directory
        info_file_path = os.path.join(session_results_dir, 'info.json')
        with open(info_file_path, 'w') as info_file:
            json.dump(model_info, info_file, indent=4)

        print(f'Model information saved to {info_file_path}')

        # Call the train function
        train(args.model_name, args.train_data_dir, args.epochs, args.batch_size, 
            args.learning_rate, args.device, args.save_interval, args.patience, 
            args.train_split, session_results_dir)

        test(session_dir=session_results_dir, test_data_dir=args.train_data_dir, 
            batch_size=args.batch_size, device=args.device)
    
    else:
        test(session_dir=args.session_path, test_data_dir=args.train_data_dir, 
            batch_size=args.batch_size, device=args.device)

if __name__ == "__main__":
    main()
