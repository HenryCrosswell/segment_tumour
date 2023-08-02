import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model import CNNModel, TumourSegmentationTrainer, test_model
from functions import BrainTumourDataset
from pathlib import Path
import os

def test_CNNModel():
    num_classes = 4
    dropout_rate = 0.2

    model = CNNModel(num_classes=num_classes, dropout_rate=dropout_rate)

    # Test the model architecture
    x = torch.randn(1, 1, 224, 224)
    output = model(x)
    assert output.size() == (1, num_classes)

    # Test the dropout rate
    for module in model.features:
        if isinstance(module, nn.Dropout2d):
            assert module.p == dropout_rate

def test_TumourSegmentationTrainer():
    
    current_working_directory = Path.cwd()
    sample_data_folder = current_working_directory.joinpath(Path('tests/test_data/Training'))
    # Sample hyperparameters for testing
    batch_size = 32
    dropout_rate = 0.2
    learning_rate = 0.001
    num_epochs = 2

    # Create dummy datasets for training, validation, and testing
    transform = ToTensor()
    train_dataset = BrainTumourDataset(sample_data_folder, transform=transform)
    val_dataset = BrainTumourDataset(sample_data_folder, transform=transform)
    test_dataset = BrainTumourDataset(sample_data_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the model, criterion, and optimizer
    num_classes = 4
    model = CNNModel(num_classes=num_classes, dropout_rate=dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create the trainer and perform training
    trainer = TumourSegmentationTrainer(model, train_loader, val_loader, test_loader, criterion, optimizer, 'cpu')
    trainer.train(num_epochs=num_epochs)

    # Test the model on the test dataset
    accuracy = test_model(trainer.model, test_loader, 'cpu')

    # Ensure that the test accuracy is within a reasonable range (0 to 1)
    assert 0.0 <= accuracy <= 1.0