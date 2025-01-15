# CUDA Neural Network for MNIST Classification

## Project Overview
This project implements a neural network from scratch using CUDA C++ to classify handwritten digits from the MNIST dataset. The implementation focuses on performance and parallel processing capabilities of GPUs while maintaining a clear and modular codebase.

## Technical Specifications

### Architecture
- **Framework**: Custom CUDA C++ implementation (no deep learning frameworks used)
- **Network Structure**: 
  - Input Layer: 784 neurons (28x28 pixel images)
  - Hidden Layer: 512 neurons with ReLU activation
  - Output Layer: 10 neurons with Softmax activation
- **Training Parameters**:
  - Batch Size: 32
  - Learning Rate: 0.001 with decay (0.95 every 5 epochs)
  - Epochs: 20
  - Loss Function: Cross-entropy
  - Optimization: Momentum-based gradient descent

### Performance
- Training Accuracy: 84.48%
- Test Accuracy: 84.86%
- Training Time: ~2-3 minutes on modern GPUs

### Technologies Used
- CUDA C++ for neural network implementation
- Python for data preparation and running scripts
- NVIDIA CUDA Toolkit (>= 11.0)
- C++11 standard features
- PyTorch (for data downloading only)

## Prerequisites

### Required Skills
- Strong C++ programming knowledge
- Understanding of CUDA programming model
- Basic Python programming
- Neural network and deep learning concepts
- Understanding of parallel computing principles

### System Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0 or higher
- Python 3.7+
- Linux/Unix environment (recommended)
- CMake (optional, for building)

### Required Python Packages
```bash
pip install torch torchvision
```

## Project Structure
```
.
├── include/
│   ├── neural_network.h    # Neural network declarations
│   ├── data_loader.h       # Data loading utilities
│   └── utils.h            # Helper functions
├── src/
│   ├── main.cu            # Main training loop
│   ├── neural_network.cu  # Network implementation
│   └── data_loader.cu     # MNIST data loading
├── scripts/
│   ├── downloader.py      # MNIST data downloader
│   └── runner.py          # Training script
├── data/                  # Downloaded MNIST data
├── build/                 # Build directory
└── README.md
```

## Implementation Details

### Key Features
1. **Custom CUDA Kernels**:
   - Matrix multiplication for layer operations
   - Parallel softmax computation
   - Efficient backpropagation
   - Optimized memory access patterns

2. **Memory Management**:
   - Efficient GPU memory allocation
   - Proper cleanup and resource management
   - Batched processing for large datasets

3. **Training Optimizations**:
   - Learning rate decay
   - Momentum-based updates
   - Batch normalization
   - Numerical stability improvements

## Usage

### 1. Clone the Repository
```bash
git clone <repository-url>
cd cuda-mnist-nn
```

### 2. Download MNIST Data
```bash
python scripts/downloader.py
```

### 3. Build and Run
```bash
# Option 1: Using runner script (recommended)
python scripts/runner.py

# Option 2: Manual compilation and running
mkdir build
cd build
nvcc -O3 ../src/*.cu -I../include -o mnist_train
./mnist_train ../data/mnist_train.csv ../data/mnist_test.csv
```

### Command Line Options
```bash
python scripts/runner.py --help
  --skip-download  Skip downloading MNIST data if already present
  --skip-compile   Skip compiling CUDA code if already compiled
```

## Results and Performance
The network achieves:
- ~84.86% test accuracy after 20 epochs
- Stable training with gradual accuracy improvement
- Good generalization (small gap between train and test accuracy)
- Learning rate decay helps stabilize later training stages

Training Progress Example:
```
Epoch 0: Loss=2.2803, Train Acc=27.35%, Test Acc=27.71%
...
Epoch 19: Loss=0.5507, Train Acc=84.48%, Test Acc=84.86%
```

## Future Improvements
- Multi-GPU support
- Additional optimization techniques
- Configurable network architecture
- Checkpoint saving and loading
- Real-time training visualization
- Support for other datasets

## Acknowledgments
- NVIDIA for CUDA toolkit and documentation
- SIBOEHM's amazing anthropic blog post on matmul kernels! (https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/10_kernel_warptiling.cuh)
- Claude for helping me write this readme (shhhhh)

## Contact
For questions or feedback, please open an issue in the repository.