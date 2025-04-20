#!/usr/bin/env python3
"""
Launcher script for AI Scientists learning framework.

This script provides a simplified interface to run either train.py (simultaneous learning)
or continued.py (sequential learning) with appropriate configuration options.
"""

import argparse
import subprocess
import os
import sys
import time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Launcher for AI Scientists learning framework')
    
    # Basic configuration
    parser.add_argument('--mode', type=str, choices=['simultaneous', 'continued'], 
                        default='simultaneous', help='Training mode: simultaneous or continued')
    parser.add_argument('--systems', type=int, default=4, 
                        help='Number of physical systems to use')
    parser.add_argument('--synthetic', type=int, default=0, 
                        help='Number of synthetic systems to include')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    # Training parameters
    parser.add_argument('--epochs1', type=int, default=100, 
                        help='Number of warmup epochs')
    parser.add_argument('--epochs2', type=int, 
                        help='Number of main training epochs (default depends on mode)')
    parser.add_argument('--batch_size', type=int, default=512, 
                        help='Training batch size')
    
    # Model parameters
    parser.add_argument('--width', type=int, default=20, 
                        help='Width of neural network layers')
    parser.add_argument('--dimension', type=int, default=1, 
                        help='Dimension of the problem')
    
    # Regularization
    parser.add_argument('--l1', type=float, default=0.01, 
                        help='Weight for L1 regularization')
    parser.add_argument('--dl', type=float, default=0.5, 
                        help='Weight for diagonal regularization')
    
    # Output configuration
    parser.add_argument('--save_dir', type=str, 
                        default='/home/gridsan/xfu/ai_scientists/weights', 
                        help='Directory to save weights')
    parser.add_argument('--pt_dir', type=str, 
                        default='/home/gridsan/xfu/ai_scientists/pt', 
                        help='Directory to save PyTorch models')
    parser.add_argument('--run_name', type=str, required=True, 
                        help='Name for this experiment run')
    
    # GPU settings
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU device ID to use (-1 for CPU)')
    
    return parser.parse_args()

def main():
    """Main function to run the appropriate training script."""
    args = parse_args()
    
    # Set default epochs based on mode if not specified
    if args.epochs2 is None:
        if args.mode == 'simultaneous':
            args.epochs2 = 50000
        else:  # continued
            args.epochs2 = 10000
    
    # Determine which script to run
    script = 'train.py' if args.mode == 'simultaneous' else 'continued.py'
    
    # Set environment variables for GPU
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Build command with all arguments
    cmd = [
        sys.executable, script,
        '--total_systems', str(args.systems),
        '--n_synthetic', str(args.synthetic),
        '--seed', str(args.seed),
        '--epochs_1', str(args.epochs1),
        '--epochs_2', str(args.epochs2),
        '--batch_size', str(args.batch_size),
        '--width', str(args.width),
        '--dimension', str(args.dimension),
        '--l1_weight', str(args.l1),
        '--dl_weight', str(args.dl),
        '--save_dir', args.save_dir,
        '--pt_dir', args.pt_dir,
        '--run_name', args.run_name
    ]
    
    # Print command for reference
    cmd_str = ' '.join(cmd)
    print(f"Running command: {cmd_str}")
    
    # Run the command
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"Training completed successfully in {elapsed:.2f} seconds ({elapsed/3600:.2f} hours)")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 