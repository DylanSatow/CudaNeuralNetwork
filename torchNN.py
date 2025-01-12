import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Hyperparameters
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 10

torch.manual_seed(42)

class MLP(nn.Module):
    
    def __init__(self, input_size=784, num_classes=10, hidden_size=256):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.
        )
    

# Data loading and transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Allocate the tensors of the right size
train_data = torch.zeros(len(train_dataset), 1, 28, 28)
train_labels = torch.zeros(len(train_dataset), dtype=torch.long)
train_data = torch.zeros(len(test_dataset), 1, 28, 28)
test_labels = torch.zeros(len(test_dataset), dtype=torch.long)

