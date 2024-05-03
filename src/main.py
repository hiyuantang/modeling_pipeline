import os
os.makedirs('./results', exist_ok=True)
import argparse
import torch
import json
from train import train
from test import test
from utils import *

def main():
    parser = argparse.ArgumentParser(description='Modeling Pipeline.')
    parser.add_argument('--model_name', type=str, default='vit_b_16', help='Name of the model')
    parser.add_argument('--train_data_dir', type=str, default='E:/depth_None_True/trainset.pkl', help='Directory of the training data')
    parser.add_argument('--test_data_dir', type=str, default='E:/depth_None_True/testset.pkl', help='Directory of the testing data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='Drop out rate')
    parser.add_argument('--pre_trained', type=bool, default=True, help='Pre-trained weights for the model')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], help='Device to train and test on')
    parser.add_argument('--save_interval', type=int, default=-1, help='Interval to save the model')
    parser.add_argument('--patience', type=int, default=30, help='Patience for early stopping')
    parser.add_argument('--train_split', type=float, default=0.9, help='Train split ratio')
    parser.add_argument('--test_only', type=bool, default=False, help='Make model inference only without training')
    parser.add_argument('--session_path', type=str, required=False, help='Path for the session for testing')
    args = parser.parse_args()

    if args.model_name.startswith('baseline'):
        args.pre_trained = False
    
    # Define training and testing device automatically (Removed TPU support)
    if args.device == 'auto': 
        if torch.cuda.is_available():
            args.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            args.device = torch.device('mps')
        # elif "TPU_ACCELERATOR_TYPE" in os.environ:
        #     import torch_xla.core.xla_model as xm
        #     args.device = xm.xla_device()
        else:
            args.device = torch.device('cpu')
    else:
        pass
    
    print(f"Using device: {args.device}")

    # Do both training and testing
    if args.test_only==False:

        # Generate a unique hash for the current training session
        session_results_dir = generate_unique_hash()
        os.makedirs(session_results_dir, exist_ok=True)

        model_temp, model_summary = get_pretrained_model(args.model_name, num_classes=1, 
                                                             drop_rate=args.drop_rate, batch_size=args.batch_size, 
                                                             pretrained=args.pre_trained)
        
        total_params = count_parameters(model_temp)
        del model_temp

        model_summary_file_path = os.path.join(session_results_dir, 'model_summary.txt')
        with open(model_summary_file_path, 'w') as summary_file:
            summary_file.write(model_summary)

        # Create a dictionary with model information
        model_info = {
            'id': session_results_dir, 
            'model_name': args.model_name,
            'train_data_dir': args.train_data_dir,
            'test_data_dir': args.test_data_dir, 
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'drop_rate': args.drop_rate, 
            'pre_trained': args.pre_trained,
            'device': str(args.device),
            'save_interval': args.save_interval,
            'patience': args.patience,
            'train_split': args.train_split, 
            'total_params': total_params
        }

        # Save the model information to info.json within the session directory
        info_file_path = os.path.join(session_results_dir, 'info.json')
        with open(info_file_path, 'w') as info_file:
            json.dump(model_info, info_file, indent=4)

        print(f'Model information saved to {info_file_path}')

        # Call the train function
        train(args.model_name, args.train_data_dir, args.epochs, args.batch_size, 
            args.learning_rate, args.drop_rate, args.pre_trained, args.device, args.save_interval, args.patience, 
            args.train_split, session_results_dir)

        test(session_dir=session_results_dir, test_data_dir=args.train_data_dir, 
            batch_size=args.batch_size, drop_rate=args.drop_rate, device=args.device)
    
    # Do only testing without training
    else:
        test(session_dir=args.session_path, test_data_dir=args.test_data_dir, 
            batch_size=args.batch_size, device=args.device)

if __name__ == "__main__":
    main()
