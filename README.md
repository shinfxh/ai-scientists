# MASS Training

This repository contains the codebase for the paper Do Two AI Scientists Agree? (https://arxiv.org/abs/2504.02822)

## Overview

As presented in the paper, we use continued training by exposing the model to new systems one by one. train.py is included as an alternative exploratory direction but is not actively maintained. 

1. **Simultaneous Training** (`train.py`): Trains the model on multiple physical systems simultaneously.
2. **Continued Learning** (`continued.py`): Exposes the model to physical systems one by one, allowing it to build cumulative knowledge.


## Key Components

### Model Architecture (`model_utils.py`)

The current implementation is a 6-layer MLP with a GELU activation. Computation of higher-order derivatives is highly unstable and we introduce a small epsilon term and pseudo-inverse calculation to stabilize it numerically.

### Physics Systems (`systems.py`)

The repository contains implementations of various physical systems for training and evaluation:

- Classical systems (harmonic oscillator, pendulum, etc.)
- Non-linear systems (Morse potential, quartic potential, etc.)
- Relativistic systems
- Synthetic systems with custom equation forms

The paper makes use of 6 key systems of concern: _classical, _pendulum, _kepler, _relativistic, _synthetic_sin, _synthetic_sin2, _synthetic_exp.


## Installation

### Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- pandas
- timm

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Using the Launcher Script (Recommended)

The easiest way to run experiments is with the `run.py` launcher script:

```bash
python run.py --mode continued --systems 4 --run_name "sequential_learning"
```

Key launcher arguments:
- `--mode`: Choose between `simultaneous` or `continued` training
- `--systems`: Number of physical systems to use
- `--synthetic`: Number of synthetic systems to include
- `--epochs1`: Number of warmup epochs
- `--epochs2`: Number of main training epochs
- `--run_name`: Required unique name for the experiment
- `--gpu`: GPU ID to use (-1 for CPU)

Run with `--help` to see all available options:

```bash
python run.py --help
```

### Direct Script Usage

Alternatively, you can run the training scripts directly:

#### Training with Continued Learning

Train the model by exposing it to systems one by one:

```bash
python continued.py --total_systems 4 --seed 42 --epochs_1 100 --epochs_2 10000
```

### Important Parameters

Both training scripts support the following parameters:

- `--seed`: Random seed for reproducibility
- `--total_systems`: Number of physical systems to use
- `--n_synthetic`: Number of synthetic systems to include
- `--width`: Width of the neural network layers
- `--dimension`: Dimension of the problem
- `--epochs_1`: Number of warmup epochs
- `--epochs_2`: Number of main training epochs
- `--lr`: Learning rate
- `--batch_size`: Mini-batch size
- `--l1_weight`: Weight for L1 regularization
- `--dl_weight`: Weight for diagonal regularization
- `--save_dir`: Directory to save weights and results
- `--run_name`: Name of the experiment run

Run with `--help` to see all available options:

```bash
python train.py --help
python continued.py --help
```

## Output and Evaluation

The training process saves:

- Model weights at regular intervals
- Final model weights
- EMA (Exponential Moving Average) model weights
- Activation values for analysis
- Loss curves and evaluation metrics

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Citation

If you use this code in your research, please cite:

```
@misc{fu2025aiscientistsagree,
      title={Do Two AI Scientists Agree?}, 
      author={Xinghong Fu and Ziming Liu and Max Tegmark},
      year={2025},
      eprint={2504.02822},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.02822}, 
}
```
