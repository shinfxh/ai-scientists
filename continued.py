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

seed = 42
total_systems = 4
n_synthetic = 0

if not is_notebook():
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    if len(sys.argv) > 2:
        total_systems = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_synthetic = int(sys.argv[3])

np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

w = 20
d = 1
dnn = DNN(d=d, w=w, ts=total_systems)
dnn = dnn.to(device)
model_ema = ModelEmaV2(dnn, decay=0.99, device=device)
epochs_1 = 100
epochs_2 = 10000
n_train = 10000
d = 1
coeffs = [[0,2,1,1]] # s = y^2 + xy
Xs, Ys = generate_data(n_train, d, total_systems, device=device, n_synthetic=n_synthetic, coeffs=coeffs)
Xs_test, Ys_test = generate_data(10000, 1, total_systems, device=device, n_synthetic=n_synthetic, coeffs=coeffs)
batch_size = 512
l = 0.01
curr = l
dl = 0.5
ent = 0.
window = 10

# Define save directories
save_dir = f'/home/gridsan/xfu/ai_scientists/weights/systems_{total_systems}_1501a/seed_{seed}/'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

pt_dir = f'/home/gridsan/xfu/ai_scientists/pt/systems_{total_systems}_1501a/seed_{seed}/'
if not os.path.isdir(pt_dir):
    os.makedirs(pt_dir)

loss_arr = np.array([])
ema_loss_arr = np.array([])

reg_dict = {}

reg_dict['b'] = [dl * dnn.b0.cpu().detach().numpy()]
reg_dict['weights'] = []

epoch_per_eval = 500

activations = [[] for _ in range(total_systems)]

def reg(model, i):
    last_weights = model.lf.weight[model.d:, :]
    
    activation = last_weights * dnn.vectors
    activations[i] = activation.detach().cpu().numpy()
    reg_last = torch.norm(activation, p=1)/172 * l

    # mask = torch.ones_like(last_weights).detach()
    # mask[:, 28:] *= 1.5
    # last_weights = last_weights * mask #increase regularization for 3 terms
    
    reg_last += torch.norm(last_weights, p=1)/172 * l * 10

    # weights_normalized = torch.abs(last_weights)
    # reg_entropy = -torch.sum(weights_normalized * torch.log(weights_normalized)) * ent
    
    b = getattr(dnn, f'b{i}')
    reg_b = torch.abs(b) * dl
    return reg_last, reg_b


def loss_fn(model, inputs, labels, i, train=True, threshold=1e1):
    pred = model.predict(inputs, i, train=train, threshold=threshold)  
    loss_qtt = torch.norm(pred[:, d:]-labels[:, d:], p=2)
    loss_sum = (loss_qtt) / torch.sqrt(torch.tensor(inputs.shape[0], dtype=torch.float)) 
    return loss_sum

def eval_fn(model, Xs, Ys, total_systems):
    model.eval()
    max_loss = -1e99
    for i in range(total_systems):
        X = Xs[i]
        Y = Ys[i]
        loss = loss_fn(model, X, Y, i, train=False, threshold=1e1)
        max_loss = max(max_loss, loss)
    return max_loss
        
def train_loop(start_epoch, end_epoch, loss_arr, ema_loss_arr, total_systems):
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
            loss += loss_fn(dnn, inputs, labels, i) 
            reg_last_i, reg_b_i = reg(dnn, i) # run this after the forward pass

            # Normalize by total number of systems
            loss = loss / total_systems
            reg_last_i = reg_last_i / total_systems
            reg_b_i = reg_b_i / total_systems

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
            eval_loss = eval_fn(dnn, Xs_test, Ys_test, total_systems)
            curr_max_loss = np.max(loss_arr[-window:])
            print('Epoch {}: Train Loss = {}, Eval Loss = {}, Runtime = {:.2f}s, Reg_b: {}, Reg_last: {}'\
                  .format(epoch, np.mean(loss_arr[-100:]), eval_loss, time.time() - start_time, 
                          np.max(reg_dict['b'][-window:]), np.max(reg_dict['weights'][-window:])))
            ema_loss = eval_fn(model_ema.module, Xs_test, Ys_test, total_systems)
            ema_loss_arr = np.append(ema_loss_arr, ema_loss.item())
            print('EMA eval loss: {}'.format(ema_loss))
            start_time = time.time()

            weights = dnn.lf.cpu().weight.detach().numpy()[1]
            pd.DataFrame(weights).to_csv(os.path.join(save_dir, f'step_{epoch}.csv'), index=False)
    
    return loss_arr, ema_loss_arr

scheduler = None

start_epoch = 0
end_epoch = 0

for t_curr in range(1, total_systems+1, 1):
    optimizer = optim.AdamW(dnn.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.7, 0.8))
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=epochs_1)
    start_epoch = end_epoch + 1
    end_epoch += epochs_1
    loss_arr, ema_loss_arr = train_loop(start_epoch, end_epoch, loss_arr, ema_loss_arr, t_curr)
    
    optimizer = optim.AdamW(dnn.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.7, 0.8))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_2)
    start_epoch = end_epoch + 1
    end_epoch += epochs_2
    loss_arr, ema_loss_arr = train_loop(start_epoch, end_epoch, loss_arr, ema_loss_arr, t_curr) 

    for i, act_i in enumerate(activations):
        act_df = pd.DataFrame(act_i)
        this_save_path = os.path.join(save_dir, f'system_{t_curr}_activations_{i}.csv')
        act_df.to_csv(this_save_path, index=False)
        
    # Save the model weights at the end of training using pt_dir
    model_path = os.path.join(pt_dir, f'model_{t_curr}.pt')
    torch.save(dnn.state_dict(), model_path)
    
    # Save the EMA model’s weights
    ema_model_path = os.path.join(pt_dir, f'model_{t_curr}.pt')
    torch.save(model_ema.module.state_dict(), ema_model_path)

# Final Evaluation and Saving
model = model_ema.module
model.eval()
i = 0
Xs_test, Ys_test = generate_data(10000, 1, total_systems, device=device, n_synthetic=n_synthetic, coeffs=coeffs)
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

# Save final weights and metrics using save_dir
pd.DataFrame(weights).to_csv(os.path.join(save_dir, 'final.csv'), index=False)
pd.DataFrame(ema_loss_arr).to_csv(os.path.join(save_dir, 'ema_loss.csv'), index=False)

for i, act_i in enumerate(activations):
    act_df = pd.DataFrame(act_i)
    final_act_path = os.path.join(save_dir, f'final_activations_{i}.csv')
    act_df.to_csv(final_act_path, index=False)

# Save the model weights at the end of training using pt_dir
final_model_path = os.path.join(pt_dir, 'final_model.pt')
torch.save(dnn.state_dict(), final_model_path)

# Save the EMA model’s weights
final_ema_model_path = os.path.join(pt_dir, 'final_model_ema.pt')
torch.save(model_ema.module.state_dict(), final_ema_model_path)
