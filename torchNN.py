import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights similarly to our numpy version
        torch.nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        torch.nn.init.uniform_(self.fc2.weight, -0.1, 0.1)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_mnist_data(num_train_samples=10000, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)
    
    # Take subset of training data if specified
    if num_train_samples:
        train_dataset = Subset(train_dataset, range(num_train_samples))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train(model, train_loader, test_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if i % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, '
                      f'Batch {i}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        test_loss /= len(test_loader)
        accuracy = correct / total
        
        print(f'Epoch {epoch+1} - '
              f'Avg Loss: {total_loss/num_batches:.4f}, '
              f'Test Loss: {test_loss:.4f}, '
              f'Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_mnist_data(
        num_train_samples=10000, 
        batch_size=32
    )
    
    # Create model
    model = SimpleNN(
        input_size=784,    # 28x28 pixels
        hidden_size=256,
        output_size=10     # 10 digits
    )
    
    # Train model
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=3,
        learning_rate=0.01,
        device=device
    )