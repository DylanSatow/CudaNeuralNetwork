import numpy as np
from torchvision import datasets, transforms

def load_mnist_data(num_train_samples=10000):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)

    X_train = mnist_train.data.numpy()[:num_train_samples]
    X_train = X_train.reshape(num_train_samples, 784) / 255.0  # Flatten immediately
    y_train = mnist_train.targets.numpy()[:num_train_samples]
    
    num_test = len(mnist_test)
    X_test = mnist_test.data.numpy().reshape(num_test, 784) / 255.0
    y_test = mnist_test.targets.numpy()
    
    return X_train, y_train, X_test, y_test

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize with smaller weights for better numerical stability
        self.weights1 = np.random.uniform(-0.1, 0.1, (input_size, hidden_size))
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.uniform(-0.1, 0.1, (hidden_size, output_size))
        self.bias2 = np.zeros((1, output_size))
        
        # Store dimensions for convenience
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x):
        # First layer
        self.input = x  # Store for backprop
        self.layer1 = x @ self.weights1 + self.bias1
        
        # ReLU activation
        self.relu_output = relu(self.layer1)
        
        # Second layer
        self.layer2 = self.relu_output @ self.weights2 + self.bias2
        
        return self.layer2

    def compute_loss(self, output, targets):
        """Compute cross entropy loss"""
        batch_size = output.shape[0]
        self.probs = softmax(output)
        
        # Compute cross entropy loss
        correct_logprobs = -np.log(self.probs[range(batch_size), targets] + 1e-10)
        loss = np.sum(correct_logprobs) / batch_size
        return loss

    def backward(self, targets):
        batch_size = self.input.shape[0]
        
        # Gradient of softmax with cross-entropy
        grad_output = self.probs.copy()
        grad_output[range(batch_size), targets] -= 1
        grad_output = grad_output / batch_size
        
        # Gradient for second layer
        self.grad_weights2 = self.relu_output.T @ grad_output
        self.grad_bias2 = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient through ReLU
        grad_hidden = grad_output @ self.weights2.T
        grad_hidden *= relu_derivative(self.layer1)
        
        # Gradient for first layer
        self.grad_weights1 = self.input.T @ grad_hidden
        self.grad_bias1 = np.sum(grad_hidden, axis=0, keepdims=True)

    def update(self, learning_rate):
        """Update weights and biases using computed gradients"""
        self.weights1 -= learning_rate * self.grad_weights1
        self.bias1 -= learning_rate * self.grad_bias1
        self.weights2 -= learning_rate * self.grad_weights2
        self.bias2 -= learning_rate * self.grad_bias2

def train(model, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate):
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Shuffle training data
        perm = np.random.permutation(n_samples)
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        total_loss = 0.0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            # Get batch
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            # Forward pass
            output = model.forward(batch_X)
            loss = model.compute_loss(output, batch_y)
            total_loss += loss
            
            # Backward pass and update
            model.backward(batch_y)
            model.update(learning_rate)
            
            if i % 100 == 0:
                print(f"Batch {i}/{n_batches}, Loss: {loss:.4f}")
        
        # Evaluate on test set
        test_output = model.forward(X_test)
        test_loss = model.compute_loss(test_output, y_test)
        predictions = np.argmax(test_output, axis=1)
        accuracy = np.mean(predictions == y_test)
        
        print(f"Epoch {epoch+1} - Avg Loss: {total_loss/n_batches:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_mnist_data(10000)
    
    # Create and train model
    model = NeuralNetwork(784, 256, 10)
    train(model, X_train, y_train, X_test, y_test, 
          batch_size=32, epochs=3, learning_rate=0.01)