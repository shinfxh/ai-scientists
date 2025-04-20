"""
Physics systems module for generating training data based on various physical systems.

This module provides a collection of functions that generate synthetic data from various
physical systems, including classical, relativistic, and custom synthetic systems.
Each function represents a specific physical system with its corresponding equations of motion.
"""

import torch
from torch.autograd import Variable

# classical (1D harmonic oscillator)
def _classical(n, d, omega=1.):
    """
    Generate data for a classical harmonic oscillator system.
    
    Args:
        n (int): Number of data points to generate
        d (int): Dimension of the problem
        omega (float): Angular frequency of the oscillator
        
    Returns:
        tuple: (X, Y) where X contains input states and Y contains derivatives
    """
    canonical = torch.rand(n, 2*d) * 2 - 1 # uniform sampling, ensures more even coverage of phase space (better for plotting)
    X = canonical
    Y = torch.zeros_like(X)
    Y[:, :d] = X[:, d:]
    Y[:, d:] = -X[:, :d] * omega**2
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

# pendulum (1D simple pendulum)
def _pendulum(n, d, omega=1.):
    canonical = torch.rand(n, 2*d) * 2 - 1 # uniform sampling, ensures more even coverage of phase space (better for plotting)
    X = torch.pi * canonical #theta and theta_dot
    Y = torch.zeros_like(X)
    Y[:, :d] = X[:, d:]
    Y[:, d:] = -torch.sin(X[:, :d]) * omega**2
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

# # spherical pendulum (pendulum parameterized by spherical coordinates wiht fixed length) #TODO: this is a problem for 2D
# def _spherical_pendulum(n, d, omega=1.):
#     canonical = torch.rand(n, 2*d) * 2 - 1 # uniform sampling, ensures more even coverage of phase space (better for plotting)
#     X = torch.pi * canonical #theta and theta_dot
#     Y = torch.zeros_like(X)
#     Y[:, :d] = X[:, d:]
#     Y[:, d:] = -torch.sin(X[:, :d]) * omega**2
#     X = Variable(X, requires_grad=True)
#     Y = Variable(Y, requires_grad=True)
#     return X, Y

# cosh potential
def _cosh(n, d, omega=1.):
    canonical = torch.rand(n, 2*d) * 2 - 1 # uniform sampling, ensures more even coverage of phase space (better for plotting)
    canonical = canonical * 2
    X = canonical
    Y = torch.zeros_like(X)
    Y[:, :d] = X[:, d:]
    Y[:, d:] = -torch.sinh(X[:, :d]) * omega**2
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

# morse potential
def _morse(n, d, omega=1.):
    canonical = torch.rand(n, 2*d) * 2 - 1 # uniform sampling, ensures more even coverage of phase space (better for plotting)
    canonical = canonical * 2
    X = canonical
    x = X[:, :d]
    y = X[:, d:]
    Y = torch.zeros_like(X)
    Y[:, :d] = X[:, d:]
    Y[:, d:] = -x * torch.exp(-x**2) * omega**2
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

# relativistic morse potential
def _relativistic_morse(n, d, omega=1.):
    canonical = torch.rand(n, 2*d) * 2 - 1 # uniform sampling, ensures more even coverage of phase space (better for plotting)
    # canonical = canonical * 2
    canonical[:, d:] *= 0.9
    X = canonical
    x = X[:, :d]
    y = X[:, d:]
    Y = torch.zeros_like(X)
    Y[:, :d] = X[:, d:]
    Y[:, d:] = -x * torch.exp(-x**2) * (1-y**2)**1.5 * omega**2
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

#quartic potential
def _quartic(n, d, omega=1.):
    canonical = torch.rand(n, 2*d) * 2 - 1 # uniform sampling, ensures more even coverage of phase space (better for plotting)
    X = canonical
    Y = torch.zeros_like(X)
    Y[:, :d] = X[:, d:]
    Y[:, d:] = -X[:, :d]**3 * omega**2
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y 


# relativistic quartic potential
def _relativistic_quartic(n, d, omega=1.):
    canonical = torch.rand(n, 2*d) * 2 - 1 # uniform sampling, ensures more even coverage of phase space (better for plotting)
    canonical[:, d:] *= 0.99
    X = canonical
    Y = torch.zeros_like(X)
    Y[:, :d] = X[:, d:]
    Y[:, d:] = -X[:, :d]**3 * (1 - X[:, d:]**2)**(1.5) * omega**1
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y 


# gravitational (inverse square) potential
def _kepler(n, d, omega=1.):
    min_val = 0.5
    max_val = 2
    canonical = torch.rand(n//2, 2*d)
    canonical[:, :d] = (max_val - min_val) * canonical[:, :d] + min_val
    canonical_negative = torch.rand(n//2, 2 * d)
    canonical_negative[:, :d] = (min_val - max_val) * canonical_negative[:, :d] - min_val
    canonical = torch.cat([canonical, canonical_negative], axis=0)
    X = canonical
    x = X[:, :d]
    y = X[:, d:]
    Y = torch.zeros_like(X)
    Y[:, :d] = X[:, d:]
    Y[:, d:] = -x/(x.abs()**3) * omega**2
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

# L = x^2 v^2
def _x2v2(n, d, omega=1.):
    min_val = 0.5
    max_val = 2
    canonical = torch.rand(n//2, 2*d)
    canonical[:, :d] = (max_val - min_val) * canonical[:, :d] + min_val
    canonical_negative = torch.rand(n//2, 2 * d)
    canonical_negative[:, :d] = (min_val - max_val) * canonical_negative[:, :d] - min_val
    canonical = torch.cat([canonical, canonical_negative], axis=0)
    X = canonical
    x = X[:, :d]
    y = X[:, d:]
    Y = torch.zeros_like(X)
    Y[:, :d] = X[:, d:]
    Y[:, d:] = -y**2/x * omega**2
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

# relativistic
def _relativistic(n, d, omega=1.):
    canonical = torch.rand(n, 2*d) * 2 - 1
    canonical[:, d:] *= 0.9
    X = canonical
    x = X[:, :d]
    y = X[:, d:]
    Y = torch.zeros_like(X)
    Y[:, :d] = y
    Y[:, d:] = -x*(1-y**2)**1.5 * omega**2
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

# relativistic cosh potential
def _relativistic_cosh(n, d, omega=1.):
    canonical = torch.rand(n, 2*d) * 2 - 1
    factor = 0.9
    canonical[:, d:] *= factor
    X = canonical
    x = X[:, :d]
    y = X[:, d:]
    Y = torch.zeros_like(X)
    Y[:, :d] = y
    Y[:, d:] = -torch.sinh(x)*(1-y**2)**1.5 * omega**2
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

# damped harmonic oscillator
def _damped(n, d, omega=1., beta=1.):
    canonical = torch.rand(n, 2*d)
    X = canonical
    x = X[:, :d]
    y = X[:, d:]
    Y = torch.zeros_like(X)
    Y[:, :d] = y
    Y[:, d:] = -omega**2 * x - beta * y
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

# synthetic lagrnagian potential with one term
def _synthetic1(n, d, a=None, b=None, omega=1.):
    canonical = torch.rand(n, 2*d) + 0.1
    X = canonical
    x = X[:, :d]
    y = X[:, d:]
    Y = torch.zeros_like(X)
    Y[:, :d] = y
    # L = x^a v^b
    if a is None:
        a = torch.randint(2, 4, ())
    if b is None:
        b = torch.randint(2, 4, ())
    Y[:, d:] = 1/(b*(b-1)*(x**a)*(y**(b-2))) * (a*x**(a-1)*y**(b) - a*b*x**(a-1)*(y**b)) * omega**2
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

def _synthetic2(n, d, a1=None, a2=None, a3=None, a4=None, omega=1.):
    canonical = torch.rand(n, 2*d) + 0.1
    X = canonical
    x = X[:, :d]
    y = X[:, d:]
    Y = torch.zeros_like(X)
    Y[:, :d] = y

    # Randomly choose a1,a2,a3,a4 if not provided
    if a1 is None:
        a1 = torch.randint(2, 4, ())
    if a2 is None:
        a2 = torch.randint(2, 4, ())
    if a3 is None:
        a3 = torch.randint(2, 4, ())
    if a4 is None:
        a4 = torch.randint(2, 4, ())

    # Potential: S = x^(a1) y^(a2) + x^(a3) y^(a4)
    # Derivatives:
    # Sx = a1 x^(a1-1) y^(a2) + a3 x^(a3-1) y^(a4)
    Sx = a1 * x**(a1-1) * y**(a2) + a3 * x**(a3-1) * y**(a4)

    # Sxy = ∂/∂y(Sx) = a1 a2 x^(a1-1) y^(a2-1) + a3 a4 x^(a3-1) y^(a4-1)
    Sxy = a1*a2 * x**(a1-1) * y**(a2-1) + a3*a4 * x**(a3-1) * y**(a4-1)

    # Syy = a2(a2-1) x^(a1) y^(a2-2) + a4(a4-1) x^(a3) y^(a4-2)
    Syy = a2*(a2-1)*x**(a1) * y**(a2-2) + a4*(a4-1)*x**(a3) * y**(a4-2)

    # Compute the requested combination:
    # Syy^{-1} Sx - Syy^{-1} Sxy y = (Sx - (Sxy * y)) / Syy
    Y[:, d:] = ((Sx - Sxy * y) / Syy) * (omega**2)

    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

def _synthetic_sin2(n, d, a1=None, a2=None, a3=None, a4=None, omega=1.):
    canonical = torch.rand(n, 2*d) + 0.1
    X = canonical
    x = X[:, :d]
    y = X[:, d:]
    Y = torch.zeros_like(X)
    Y[:, :d] = y
    # S = 1-cos(y) - cos(x) cos(y)
    Syy = torch.cos(y)
    Sx = - torch.sin(x) * torch.cos(y)
    Sxy = torch.cos(x) * torch.cos(y)
    Y[:, d:] = (Sx - Sxy * y) / Syy 
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

def _synthetic_sin(n, d, a1=None, a2=None, a3=None, a4=None, omega=1.):
    canonical = torch.rand(n, 2*d) + 0.1
    X = canonical
    x = X[:, :d]
    y = X[:, d:]
    Y = torch.zeros_like(X)
    Y[:, :d] = y
    # S = 1/2 y^2 - cos(x) cos(y)
    Sx = - torch.sin(x) * torch.cos(y)
    Sxy = torch.cos(x) * torch.cos(y)
    Y[:, d:] = Sx - Sxy * y # since Syy^{-1} = 1
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    return X, Y

def _synthetic_exp(n, d):
    # Create random inputs
    canonical = torch.rand(n, 2*d) + 0.1
    X = canonical
    x = X[:, :d]
    y = X[:, d:]
    
    # Prepare output tensor
    Y = torch.zeros_like(X)
    Y[:, :d] = y  # first half of Y is y
    
    # -------------------------------------------------------------
    # S = e^(y^2) + e^(x^2)*e^(y^2) = e^(y^2) * (1 + e^(x^2))
    # -------------------------------------------------------------
    # For consistency with the structure in _synthetic_sin2, we define:
    #
    #   Syy = d(...)    (by analogy, used as a "denominator" term)
    #   Sx  = d(...)    (by analogy, used as a "numerator" term)
    #   Sxy = d(...)    (also used in the numerator)
    #
    # then do: Y[:, d:] = (Sx - Sxy * y) / Syy

    # e^(y^2) * (1 + e^(x^2))
    S = torch.exp(y**2) * (1.0 + torch.exp(x**2))

    # By analogy, define "Syy", "Sx", "Sxy".
    # Here we pick:
    #   Syy = ∂S/∂y = 2y * e^(y^2) * (1 + e^(x^2))
    #   Sx  = ∂S/∂x = 2x * e^(x^2 + y^2)
    #   Sxy = ∂²S / (∂x∂y) = 4xy * e^(x^2 + y^2)
    #
    # (The naming follows the pattern in _synthetic_sin2, though the
    # actual PDE/derivatives might differ.)

    Syy = 2.0 * y * torch.exp(y**2) * (1.0 + torch.exp(x**2))
    Sx  = 2.0 * x * torch.exp(x**2 + y**2)
    Sxy = 4.0 * x * y * torch.exp(x**2 + y**2)

    # Fill Y's second half
    Y[:, d:] = (Sx - Sxy * y) / Syy

    # Make these Variables that require grad (if you need autograd later)
    X = Variable(X, requires_grad=True)
    Y = Variable(Y, requires_grad=True)
    
    return X, Y
    
def generate_data(n, d, ts, device=None, n_synthetic=2, coeffs=None, rand_omega=False):
    """
    Generate training data from a combination of physical systems.
    
    This function creates training data from a mix of predefined physical systems
    and optional synthetic systems. It allows specifying the number of systems,
    dimensions, and custom coefficients for synthetic systems.
    
    Args:
        n (int): Number of data points to generate per system
        d (int): Dimension of the problem
        ts (int): Total number of systems to use
        device (torch.device, optional): Device to place the tensors on (CPU or GPU)
        n_synthetic (int, optional): Number of synthetic systems to include
        coeffs (list, optional): List of coefficient lists for synthetic systems
        rand_omega (bool, optional): Whether to randomize the frequency parameter
        
    Returns:
        tuple: (Xs, Ys) where Xs is a list of input states and Ys is a list of derivatives,
        one pair per physical system
    """
    ts = ts - n_synthetic
    funcs = [_classical, _pendulum, _kepler, _relativistic, _synthetic_sin, _synthetic_sin2, _synthetic_exp]
    Xs = []
    Ys = []
    for f in funcs[:ts]:
        X, Y = f(n, d)
        if rand_omega:
            Y = torch.randn_like(X) * Y
        Xs.append(X)
        Ys.append(Y)
    for i in range(n_synthetic):
        if coeffs is not None:
            if len(coeffs[i]) == 4:
                a1, a2, a3, a4 = coeffs[i]
                X, Y = _synthetic2(n, d, a1, a2, a3, a4)
            elif len(coeffs[i]) == 2:
                a, b = coeffs[i]
                X, Y = _synthetic1(n, d, a, b)
        else:
            X, Y = _synthetic2(n, d)
        if rand_omega:
            Y = torch.randn_like(X) * Y
        Xs.append(X)
        Ys.append(Y)
    if device is not None:
        for i in range(len(Xs)):
            Xs[i] = Xs[i].to(device)
            Ys[i] = Ys[i].to(device)
    return Xs, Ys