#!/usr/bin/env python3

import os
import subprocess
import argparse
import sys
from pathlib import Path

def check_cuda_available():
    """Check if CUDA is available on the system."""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def download_data():
    """Download and prepare MNIST data using downloader.py"""
    print("Downloading and preparing MNIST data...")
    try:
        subprocess.run([sys.executable, 'downloader.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)

def compile_cuda_code():
    """Compile the CUDA neural network code."""
    print("Compiling CUDA code...")
    
    # Create build directory
    Path('build').mkdir(exist_ok=True)
    
    # Define source files
    source_files = [
        'main.cu',
        'neural_network.cu',
        'data_loader.cu'
    ]
    
    # Check if all source files exist
    for src in source_files:
        if not Path(src).exists():
            print(f"Error: Source file {src} not found!")
            sys.exit(1)
    
    # Compile command
    compile_cmd = [
        'nvcc',
        '-O3',  # Optimization level
        '-o', 'build/mnist_train',  # Output executable
        *source_files,  # Source files
        '-I./include',  # Include directory
        '--std=c++11',  # C++ standard
        '-lcudart'      # CUDA runtime library
    ]
    
    try:
        subprocess.run(compile_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        sys.exit(1)

def run_training(args):
    """Run the CUDA neural network training."""
    print("Starting neural network training...")
    
    # Construct paths
    data_dir = Path('data')
    train_path = data_dir / 'mnist_train.csv'
    test_path = data_dir / 'mnist_test.csv'
    executable = Path('build') / 'mnist_train'
    
    # Verify files exist
    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        sys.exit(1)
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        sys.exit(1)
    if not executable.exists():
        print(f"Error: Executable not found at {executable}")
        sys.exit(1)
    
    # Run the training
    try:
        cmd = [
            str(executable),
            str(train_path),
            str(test_path)
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='MNIST Neural Network Runner')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip downloading MNIST data if already present')
    parser.add_argument('--skip-compile', action='store_true',
                       help='Skip compiling CUDA code if already compiled')
    args = parser.parse_args()
    
    # Check CUDA availability
    if not check_cuda_available():
        print("Error: CUDA is not available on this system!")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('build', exist_ok=True)
    
    # Download data if needed
    if not args.skip_download:
        download_data()
    else:
        print("Skipping data download...")
    
    # Compile code if needed
    if not args.skip_compile:
        compile_cuda_code()
    else:
        print("Skipping compilation...")
    
    # Run training
    run_training(args)

if __name__ == '__main__':
    main()