import torch
import torch.nn as nn
from pathlib import Path
from functions import make_val_dataset, get_data_loaders
from model import CNNModel, TumourSegmentationTrainer, test_model
import logging
import os
import json

# Logging basics
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('training.log'),  # Save logs to a file
        logging.StreamHandler()  # Display logs in the console
    ]
)

# Working paths, used throughout main.py
current_working_directory = Path.cwd()
archive_path = current_working_directory.joinpath('archive')
validation_path = archive_path.joinpath('Validation')
test_path = archive_path.joinpath('Testing')
training_path = archive_path.joinpath('Training')

# If validation folder is not present, it moves 20% of training images for the validation set
if not os.path.exists(validation_path):
    print("Validation folder not found. Creating it...")
    make_val_dataset(archive_path, validation_ratio=0.2)
    print("Validation folder created.")
else:
    print("Validation folder already exists.")

# Initialize parameters
val_split = 0.1
num_epochs = 2
num_classes = 4
learning_rate = 0.001

train_loader, val_loader, test_loader = get_data_loaders(training_path, validation_path, test_path, batch_size=32)

# Check if the model.pth exists and load it, otherwise perform hyperparameter search
checkpoint_path = current_working_directory.joinpath('model.pth')
hyperparameters_path = current_working_directory.joinpath('hyperparameters.json')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.exists(checkpoint_path) and os.path.exists(hyperparameters_path):
    print("Existing model and hyperparameters found. Loading them...")
    # Load hyperparameters from the JSON file
    with open(hyperparameters_path, 'r') as f:
        hyperparameters = json.load(f)
    batch_size = hyperparameters["batch_size"]
    dropout_rate = hyperparameters["dropout_rate"]
    learning_rate = hyperparameters["learning_rate"]

    model = CNNModel(num_classes, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(checkpoint_path))

else:
    print("No existing model and hyperparameters found. Performing random hyperparameter search...")
    # Initialize the trainer and perform random hyperparameter search
    trainer = TumourSegmentationTrainer()
    best_hyperparameters = trainer.random_hyperparameter_search(train_loader, val_loader, test_loader)
    with open(hyperparameters_path, 'w') as f:
        json.dump(best_hyperparameters, f)

    print("Best Hyperparameters:")
    print(best_hyperparameters)

    batch_size = best_hyperparameters['batch_size']
    dropout_rate = best_hyperparameters['dropout_rate']
    learning_rate = best_hyperparameters['learning_rate']

    model = CNNModel(num_classes, dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = TumourSegmentationTrainer(model, train_loader, val_loader, test_loader, criterion, optimizer, device)
    trainer.train(num_epochs=num_epochs, log_interval=10)

    # Save the trained model
    torch.save(model.state_dict(), checkpoint_path)
    print("Model and hyperparameters saved.")

# Test the model on the test data
final_test_accuracy = test_model(model, test_loader, device)
print(f"Final Test Accuracy: {final_test_accuracy * 100:.2f}%")
