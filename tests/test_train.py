import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from main import train_model
import torch.optim as optim

def test_training():
    # Create a sample binary classification dataset
    num_samples = 32
    binary_labels = torch.randint(0, 2, (num_samples, 1)).float()
    sample_data = [(torch.randn(1, 224, 224), binary_labels) for _ in range(num_samples)]

    # Create DataLoader for the test dataset
    test_loader = torch.utils.data.DataLoader(sample_data, batch_size=4, shuffle=True)

    # Initialize the model
    model = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 2
    train_model(model, criterion, optimizer, test_loader, test_loader, num_epochs)

    # Assertion to check if the training completed without errors
    assert True
