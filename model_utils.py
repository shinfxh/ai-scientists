import torch
import numpy as np
import functools

import torch.nn as nn
from torch.autograd.functional import jacobian
from torch.func import vmap, hessian

import dim1_utils

def clip_batch(inputs, threshold=10., batch_axis=0):
    """
    Clips elements in a inputs of shape [batch_size, m] such that if abs mean over batch axis exceeds threshold, all elements are set to zero
    """
    x = torch.abs(inputs)
    x = x.mean(axis=batch_axis, keepdim=True)
    return torch.where(x < threshold, inputs, 0)
    return torch.where(x < threshold, inputs, inputs/(1+0.1*(inputs**2-threshold**2)))


# ---------------------------------------------------------
# Attention block to replace the MLP
# ---------------------------------------------------------
class AttentionBlock(nn.Module):
    """
    A simple transformer-based block to handle sequences of length 5 and hidden dimension d.
    """
    def __init__(self, d, w=200, num_heads=2, num_layers=2):
        super(AttentionBlock, self).__init__()
        self.d = d
        self.d_model = d            # We'll treat each token as dimension d
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # A single TransformerEncoder layer; you can stack multiple with TransformerEncoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=w,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        
        # final projection to scalar
        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, x):
        # x is shape (batch_size, 5*d)
        batch_size = x.size(0)
        
        # reshape into (batch_size, sequence_length=5, d)
        x = x.view(batch_size, 5, self.d)
        
        # pass through the Transformer encoder
        x = self.transformer_encoder(x)
        
        # global average pooling over the sequence dimension
        x = x.mean(dim=1)  # shape: (batch_size, d)
        
        # map to scalar
        x = self.fc(x)     # shape: (batch_size, 1)
        return x
        


class ResidualBlock(nn.Module):
    def __init__(self, w):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(w, w)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(w, w)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out + residual
        
class DNN(nn.Module):
    """
    Derivative Neural Network for discovering physical systems.
    
    This network is designed to learn the underlying structure of physical systems
    from data. It includes specialized components for computing derivatives and
    matrix operations that help capture physical relationships.
    
    Args:
        d (int, optional): Dimension of input space. Defaults to 2.
        w (int, optional): Width of hidden layers. Defaults to 200.
        ts (int, optional): Number of theory/physical systems. Defaults to 1.
    
    Attributes:
        d (int): Dimension of input space
        ts (int): Number of theory/physical systems
        mlp{i} (nn.Sequential): MLP network for each system
        b{i} (nn.Parameter): Regularization parameter for each system
        a{i} (nn.Parameter): Scale parameter for each system
        lf (nn.Linear): Final linear layer for output
        vectors (torch.Tensor): Intermediate vectors for computation
        mats (torch.Tensor): Intermediate matrices for computation
    """
    def __init__(self, d = 2, w = 200, ts=1):
        super(DNN, self).__init__()
        self.d = d
        
        self.ts = ts # number of theories        
        
        for i in range(self.ts):
            setattr(self, f'mlp{i}', nn.Sequential(
            nn.Linear(5*d, w),
            nn.GELU(),
            nn.Linear(w, w),
            nn.GELU(),
            nn.Linear(w, w),
            nn.GELU(),
            nn.Linear(w, w),
            nn.GELU(),
            nn.Linear(w, w),
            nn.GELU(),
            nn.Linear(w, 1),
            ))
            
             # Initialize b with values from a normal distribution
            b_val = torch.randn([], dtype=torch.float32) * 0.1
            a_val = torch.randn([], dtype=torch.float32) * 0.1
            setattr(self, f'b{i}', nn.Parameter(b_val))
            setattr(self, f'a{i}', nn.Parameter(a_val))
        
        self.lf = nn.Linear(172, 2, bias=False) #final linear layer

        # for storing some intermediate computation
        self.vectors = None
        self.mats = None

    
    
    def forward(self, x, t, vmapped=False, learn_square=True, learn_cross=True):
        # x is of input shape (n, 2*d)
        mlp = getattr(self, f'mlp{t}')
        a = getattr(self, f'a{t}')
        if vmapped:
            x = x.unsqueeze(0)

        if learn_square:
            square = x**2
        else:
            square = torch.zeros_like(x)
        if learn_cross:
            cross = x[:, self.d:] * x[:, :self.d]
        else:
            cross = torch.zeros_like(x[:, 0])
        x = torch.cat([x, square, cross], axis=1)
        
        self.S = mlp(x) 
        return self.S 

    
    def predict_vectors(self, x, t):
        # Compute the forward pass once
        outputs = self.forward(x, t)
        
        # Compute first-order derivatives
        first_derivatives = torch.autograd.grad(outputs, x, torch.ones_like(outputs), create_graph=True)[0]

        # Compute second-order derivatives (Hessian) by differentiating first_derivatives
        second_derivatives = torch.zeros(x.size(0), x.size(1), x.size(1), device=x.device) # (batch_size, 2*d, 2*d)
        for i in range(x.size(1)):
            # Take gradient with respect to each input dimension
            H_second_order = torch.autograd.grad(first_derivatives[:, i], x, torch.ones_like(first_derivatives[:, i]), create_graph=True)[0]
            second_derivatives[:, i, :] = H_second_order
            
        # dimension matching
        Hx = first_derivatives[:, :self.d]
        Hy = first_derivatives[:, self.d:]
        Hxx = second_derivatives[:, :self.d, :self.d].unsqueeze(0)
        Hyy = second_derivatives[:, self.d:, self.d:].unsqueeze(0)
        Hxy = second_derivatives[:, :self.d, self.d:].unsqueeze(0)
        b = getattr(self, f'b{t}')
        e = b * torch.eye(Hxx.shape[-2]).to(Hxx.device) 
        matrices = torch.cat([Hxx, Hyy, Hxy, torch.linalg.pinv(Hxx + e), torch.linalg.pinv(Hyy + e), torch.linalg.pinv(Hxy+ e)], dim=0) #to be 6x100x1x1
        self.mats = matrices
        
        vectors = torch.cat([x[:, :self.d].unsqueeze(0), x[:, self.d:].unsqueeze(0), Hx.unsqueeze(0), Hy.unsqueeze(0)], dim=0).unsqueeze(dim=3) #4x100x1x1
        products2 = torch.einsum('ijkl,mjln->imjkn', matrices, vectors) #6x4x100x1x1
        vectors2 = products2.reshape((-1,) + products2.shape[2:]) #24x100x1x1
        products3 = torch.einsum('ijkl,mjln->jim', matrices, vectors2) #100x6x24
        vectors3 = products3.reshape((products3.shape[0], -1)) #100x144
        vectors2 = vectors2.transpose(0, 1).reshape((vectors2.shape[1], vectors2.shape[0])) #100x24
        vectors = vectors.transpose(1, 0).reshape((vectors.shape[1], vectors.shape[0])) #100x4
        vectors = torch.cat([vectors, vectors2, vectors3], dim=1) #100x172
        vectors[:, dim1_utils.repeats] *= 0
        vectors[:, :2] *= 0
        return vectors

    def predict(self, x, t, train=True, threshold=10.):
        self.vectors = self.predict_vectors(x, t)
        return self.lf(self.vectors)