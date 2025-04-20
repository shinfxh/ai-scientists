import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import sys
import copy
import scipy
import argparse
from IPython.display import clear_output
from torch.autograd.functional import jacobian
import pandas as pd
from timm.utils import ModelEmaV2
from copy import deepcopy

import model_utils
import systems
import dim1_utils
from systems import generate_data
from model_utils import DNN
from optim_utils import SAM

def is_notebook() -> bool:
    """
    Determine if the code is running in a Jupyter notebook environment.
    
    Returns:
        bool: True if running in a notebook, False otherwise
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def reg(model, i, l, dl, activations):
    """
    Compute regularization terms for the model.
    
    Args:
        model: The neural network model
        i: System index
        l: L1 regularization weight
        dl: Regularization weight for b parameter
        activations: List to store activation values for each system
    
    Returns:
        tuple: (reg_last, reg_b) - Regularization terms for last layer weights and b parameter
    """
    last_weights = model.lf.weight[model.d:, :]
    
    activation = last_weights * model.vectors
    activations[i] = activation.detach().cpu().numpy()
    reg_last = torch.norm(activation, p=1)/172 * l

    # Add additional regularization for last weights
    reg_last += torch.norm(last_weights, p=1)/172 * l * 10
    
    b = getattr(model, f'b{i}')
    reg_b = torch.abs(b) * dl
    return reg_last, reg_b


def loss_fn(model, inputs, labels, i, d, train=True, threshold=1e1):
    """
    Compute the loss function for training or evaluation.
    
    Args:
        model: The neural network model
        inputs: Input data tensor
        labels: Ground truth labels tensor
        i: System index
        d: Dimension of the problem
        train: Whether this is for training (True) or evaluation (False)
        threshold: Threshold for clipping values
    
    Returns:
        torch.Tensor: The computed loss value
    """
    pred = model.predict(inputs, i, train=train, threshold=threshold)
    loss_qtt = torch.norm(pred[:, d:]-labels[:, d:], p=2)
    loss_sum = (loss_qtt) / torch.sqrt(torch.tensor(inputs.shape[0], dtype=torch.float)) 
    return loss_sum


def eval_fn(model, Xs, Ys, d):
    """
    Evaluate the model on test data.
    
    Args:
        model: The neural network model
        Xs: List of input data tensors for each system
        Ys: List of ground truth tensors for each system
        d: Dimension of the problem
    
    Returns:
        torch.Tensor: Maximum loss across all systems
    """
    model.eval()
    ts = len(Xs)
    max_loss = -1e99
    for i in range(ts):
        X = Xs[i]
        Y = Ys[i]
        loss = loss_fn(model, X, Y, i, d, train=False, threshold=1e1)
        max_loss = max(max_loss, loss)
    return max_loss


def train_loop(start_epoch, end_epoch, loss_arr, ema_loss_arr, total_systems, dnn, model_ema, 
               optimizer, scheduler, Xs, Ys, Xs_test, Ys_test, n_train, batch_size, d, l, dl, 
               reg_dict, activations, epoch_per_eval, window, save_dir):
    """
    Main training loop for the model.
    
    Args:
        start_epoch: Starting epoch number
        end_epoch: Ending epoch number
        loss_arr: Array to store loss values
        ema_loss_arr: Array to store EMA loss values
        total_systems: Number of systems to train on
        dnn: The neural network model
        model_ema: Exponential moving average model
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        Xs: List of input data tensors for each system
        Ys: List of ground truth tensors for each system
        Xs_test: List of test input data tensors
        Ys_test: List of test ground truth tensors
        n_train: Number of training examples
        batch_size: Batch size for training
        d: Dimension of the problem
        l: L1 regularization weight
        dl: Regularization weight for b parameter
        reg_dict: Dictionary to store regularization values
        activations: List to store activation values for each system
        epoch_per_eval: Number of epochs between evaluations
        window: Window size for averaging
        save_dir: Directory to save model weights
    
    Returns:
        tuple: (loss_arr, ema_loss_arr) - Updated loss arrays
    """
    start_time = time.time()
    for epoch in range(start_epoch, end_epoch + 1):
        dnn.train()
        
        loss = 0
        reg_b = 0
        reg_last = 0

        for i in range(total_systems):
            X = Xs[i]
            Y = Ys[i]
            choices = np.random.choice(n_train, batch_size, replace=False)
            inputs = X[choices]
            labels = Y[choices]
            loss += loss_fn(dnn, inputs, labels, i, d)
            reg_last_i, reg_b_i = reg(dnn, i, l, dl, activations) # run this after the forward pass
            loss += reg_last_i + reg_b_i
            reg_last += reg_last_i
            reg_b += reg_b_i

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        model_ema.update(dnn)
            
        loss_arr = np.append(loss_arr, loss.item())  # Use loss.item() to get the loss value
        if reg_b != 0:
            reg_dict['b'].append(reg_b.item())
        else:
            reg_dict['b'].append(reg_dict['b'][-1])
        reg_dict['weights'].append(reg_last.item())

        if scheduler:
            scheduler.step()
        
        
        if epoch % epoch_per_eval == 0 or epoch == 1:
            # Do evaluation
            eval_loss = eval_fn(dnn, Xs_test, Ys_test, d)
            curr_max_loss = np.max(loss_arr[-window:])
            print('Epoch {}: Train Loss = {}, Eval Loss = {}, Runtime = {:.2f}s, Reg_b: {}, Reg_last: {}'\
                  .format(epoch, np.mean(loss_arr[-100:]), eval_loss, time.time() - start_time, 
                         np.max(reg_dict['b'][-window:]), np.max(reg_dict['weights'][-window:])))
            ema_loss = eval_fn(model_ema.module, Xs_test, Ys_test, d)
            ema_loss_arr = np.append(ema_loss_arr, ema_loss.item())
            print('EMA eval loss: {}'.format(ema_loss))
            start_time = time.time()

            weights = dnn.lf.cpu().weight.detach().numpy()[1]
            pd.DataFrame(weights).to_csv(os.path.join(save_dir, f'step_{epoch}.csv'), index=False)
    
    return loss_arr, ema_loss_arr


def main(args):
    """
    Main function to run the training process.
    
    Args:
        args: Command line arguments parsed by argparse
    """
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model and parameters
    w = args.width
    d = args.dimension
    dnn = DNN(d=d, w=w, ts=args.total_systems)
    dnn = dnn.to(device)
    model_ema = ModelEmaV2(dnn, decay=args.ema_decay, device=device)
    
    # Training parameters
    epochs_1 = args.epochs_1
    epochs_2 = args.epochs_2
    epochs_3 = args.epochs_3
    n_train = args.n_train
    
    # Define synthetic coefficients if using synthetic data
    coeffs = [[0,2,1,1]] # s = y^2 + xy
    
    # Generate data
    Xs, Ys = generate_data(n_train, d, args.total_systems, device=device, 
                          n_synthetic=args.n_synthetic, coeffs=coeffs)
    Xs_test, Ys_test = generate_data(args.n_test, d, args.total_systems, 
                                    device=device, n_synthetic=args.n_synthetic, coeffs=coeffs)
    
    batch_size = args.batch_size
    l = args.l1_weight
    curr = l
    dl = args.dl_weight
    ent = args.entropy_weight
    window = args.window

    # Define save directories
    save_dir = os.path.join(args.save_dir, f'systems_{args.total_systems}_{args.run_name}/seed_{args.seed}/')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pt_dir = os.path.join(args.pt_dir, f'systems_{args.total_systems}_{args.run_name}/seed_{args.seed}/')
    if not os.path.isdir(pt_dir):
        os.makedirs(pt_dir)

    # Initialize arrays and dictionaries
    loss_arr = np.array([])
    ema_loss_arr = np.array([])

    reg_dict = {}
    reg_dict['b'] = [dl * dnn.b0.cpu().detach().numpy()]
    reg_dict['weights'] = []

    epoch_per_eval = args.epoch_per_eval
    activations = [[] for _ in range(args.total_systems)]

    # First optimizer with linear warmup
    optimizer = optim.AdamW(dnn.parameters(), lr=args.lr, weight_decay=args.weight_decay, 
                          betas=(args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=epochs_1)
    start_epoch = 1
    end_epoch = epochs_1
    loss_arr, ema_loss_arr = train_loop(start_epoch, end_epoch, loss_arr, ema_loss_arr, args.total_systems,
                                      dnn, model_ema, optimizer, scheduler, Xs, Ys, Xs_test, Ys_test,
                                      n_train, batch_size, d, l, dl, reg_dict, activations, 
                                      epoch_per_eval, window, save_dir)

    # Second optimizer with cosine annealing
    optimizer = optim.AdamW(dnn.parameters(), lr=args.lr, weight_decay=args.weight_decay, 
                          betas=(args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_2)
    start_epoch = end_epoch + 1
    end_epoch = epochs_1 + epochs_2
    loss_arr, ema_loss_arr = train_loop(start_epoch, end_epoch, loss_arr, ema_loss_arr, args.total_systems,
                                      dnn, model_ema, optimizer, scheduler, Xs, Ys, Xs_test, Ys_test,
                                      n_train, batch_size, d, l, dl, reg_dict, activations, 
                                      epoch_per_eval, window, save_dir)

    # Final Evaluation
    model = model_ema.module
    model.eval()
    i = 0
    Xs_test, Ys_test = generate_data(args.n_test, d, args.total_systems, 
                                   device=device, n_synthetic=args.n_synthetic, coeffs=coeffs)
    X_test = Xs_test[i]
    Y_test = Ys_test[i]
    pred = model.predict(X_test, i)  
    labels = Y_test
    loss_qtt = torch.norm(pred[:, d:]-labels[:, d:], p=2)
    loss_sum = (loss_qtt) / torch.sqrt(torch.tensor(len(X_test), dtype=torch.float))        
    print('Final Evaluation Loss:', loss_sum.cpu().detach().numpy())

    print('Final layer weights:')
    weights = dnn.lf.cpu().weight.detach().numpy()[1]
    for i, weight in enumerate(weights):
        print(f'Weight {i}:', weight)

    # Save final weights and metrics
    pd.DataFrame(weights).to_csv(os.path.join(save_dir, 'final.csv'), index=False)
    pd.DataFrame(ema_loss_arr).to_csv(os.path.join(save_dir, 'ema_loss.csv'), index=False)

    # Save final activations
    for idx, act_i in enumerate(activations):
        act_df = pd.DataFrame(act_i)
        final_act_path = os.path.join(save_dir, f'final_activations_{idx}.csv')
        act_df.to_csv(final_act_path, index=False)

    # Save final model weights
    final_model_path = os.path.join(pt_dir, 'final_model.pt')
    torch.save(dnn.state_dict(), final_model_path)

    # Save final EMA model weights
    final_ema_model_path = os.path.join(pt_dir, 'final_model_ema.pt')
    torch.save(model_ema.module.state_dict(), final_ema_model_path)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Training script for physics systems')
    
    # Model parameters
    parser.add_argument('--width', type=int, default=20, help='Width of the neural network')
    parser.add_argument('--dimension', type=int, default=1, help='Dimension of the problem')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='Decay rate for EMA model')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--total_systems', type=int, default=4, help='Total number of systems to use')
    parser.add_argument('--n_synthetic', type=int, default=0, help='Number of synthetic systems')
    parser.add_argument('--epochs_1', type=int, default=100, help='Number of warmup epochs')
    parser.add_argument('--epochs_2', type=int, default=50000, help='Number of main training epochs')
    parser.add_argument('--epochs_3', type=int, default=100, help='Number of fine-tuning epochs')
    parser.add_argument('--n_train', type=int, default=10000, help='Number of training examples')
    parser.add_argument('--n_test', type=int, default=10000, help='Number of test examples')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.7, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.8, help='Beta2 for Adam optimizer')
    
    # Regularization parameters
    parser.add_argument('--l1_weight', type=float, default=0.01, help='Weight for L1 regularization')
    parser.add_argument('--dl_weight', type=float, default=0.5, help='Weight for b parameter regularization')
    parser.add_argument('--entropy_weight', type=float, default=0.0, help='Weight for entropy regularization')
    
    # Evaluation and saving parameters
    parser.add_argument('--window', type=int, default=10, help='Window size for averaging')
    parser.add_argument('--epoch_per_eval', type=int, default=500, help='Number of epochs between evaluations')
    parser.add_argument('--save_dir', type=str, default='/home/gridsan/xfu/ai_scientists/weights', 
                        help='Directory to save weights')
    parser.add_argument('--pt_dir', type=str, default='/home/gridsan/xfu/ai_scientists/pt', 
                        help='Directory to save PyTorch models')
    parser.add_argument('--run_name', type=str, default='2912g', help='Name of the run')
    
    args = parser.parse_args()
    
    # Run the main function
    main(args)