#!/usr/bin/env python3

import os
import csv
import torch
import torchvision
import torchvision.transforms as transforms

def save_as_csv(images, labels, csv_path):
    """
    Saves images and labels to a CSV file. Each row of the CSV will contain:
        label, pixel_0, pixel_1, ..., pixel_783
    """
    # images shape: (N, 1, 28, 28)
    # labels shape: (N,)
    # Flatten to (N, 784)
    num_samples = images.size(0)
    flattened_images = images.view(num_samples, -1)  # shape: (N, 784)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Write header row (optional)
        header = ["label"] + [f"pixel_{i}" for i in range(784)]
        writer.writerow(header)

        # Write data rows
        for i in range(num_samples):
            label = labels[i].item()
            pixels = flattened_images[i].tolist()
            row = [label] + pixels
            writer.writerow(row)

def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Define transformations: Convert to Tensor and (optionally) normalize
    # Mean and std for MNIST are approximately 0.1307 and 0.3081
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download MNIST (train + test)
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders to iterate
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=False
    )

    # Collect all train samples into single tensors
    train_images_list = []
    train_labels_list = []
    for images, labels in train_loader:
        # images shape: (batch_size, 1, 28, 28)
        train_images_list.append(images)
        train_labels_list.append(labels)

    train_images = torch.cat(train_images_list, dim=0)  # shape: (60000, 1, 28, 28)
    train_labels = torch.cat(train_labels_list, dim=0)  # shape: (60000,)

    # Collect all test samples into single tensors
    test_images_list = []
    test_labels_list = []
    for images, labels in test_loader:
        test_images_list.append(images)
        test_labels_list.append(labels)

    test_images = torch.cat(test_images_list, dim=0)  # shape: (10000, 1, 28, 28)
    test_labels = torch.cat(test_labels_list, dim=0)  # shape: (10000,)

    # Save training set as CSV
    train_csv_path = os.path.join(data_dir, "mnist_train.csv")
    save_as_csv(train_images, train_labels, train_csv_path)

    # Save test set as CSV
    test_csv_path = os.path.join(data_dir, "mnist_test.csv")
    save_as_csv(test_images, test_labels, test_csv_path)

    print(f"MNIST CSV files saved to: {data_dir}")
    print(f"  {train_csv_path}")
    print(f"  {test_csv_path}")

if __name__ == "__main__":
    main()
