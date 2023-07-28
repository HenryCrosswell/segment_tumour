import torch
import torch.nn as nn
from pathlib import Path
from functions import make_val_dataset, CNNModel, get_data_loaders, TumourSegmentationTrainer, BrainTumourDataset
import logging
from itertools import product

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('training.log'),  # Save logs to a file
        logging.StreamHandler()  # Display logs in the console
    ]
)

# Set up paths and parameters
current_working_directory = Path.cwd()
test_path = current_working_directory.joinpath('archive', 'Testing')
training_path = current_working_directory.joinpath('archive', 'Training')
validation_path = current_working_directory.joinpath('archive', 'Validation')
data_folder = current_working_directory.joinpath('archive')

val_split = 0.1
num_epochs = 2
num_classes = 4
learning_rate = 0.001


batch_sizes = [16, 32, 64]
dropout_rates = [0.2, 0.4, 0.6]

hyperparameter_combinations = []
for batch_size in batch_sizes:
    for dropout_rate in dropout_rates:
        combination = {'batch_size': batch_size, 'dropout_rate': dropout_rate}
        hyperparameter_combinations.append(combination)

best_batch_size = None
best_dropout_rate = None
best_validation_loss = float('inf')

# Grid search for hyperparameter tuning
for combination in hyperparameter_combinations:
    batch_size = combination['batch_size']
    dropout_rate = combination['dropout_rate']

    # Step 2: Prepare data loaders with the specified batch size
    train_loader, val_loader, test_loader = get_data_loaders(training_path, validation_path, test_path, batch_size=batch_size)


    # Step 3: Define the model with the specified dropout rate
    model = CNNModel(num_classes, dropout_rate)

    # Step 4: Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Step 5: Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Step 6: Initialize the trainer and start training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = TumourSegmentationTrainer(model, train_loader, val_loader, test_loader, criterion, optimizer, device)
    trainer.train(num_epochs=num_epochs, log_interval=10)

    # Step 7: Test the model on the test data
    validation_loss = trainer.evaluate(val_loader)

    # Print out the validation loss for each hyperparameter combination
    print(f"Batch Size: {batch_size}, Dropout Rate: {dropout_rate}, Validation Loss: {validation_loss}")

    # Update the best hyperparameters individually if the current hyperparameter performed better
    if validation_loss < best_validation_loss:
        best_batch_size = batch_size
        best_dropout_rate = dropout_rate

    best_validation_loss = min(best_validation_loss, validation_loss)

# Print the best hyperparameters and corresponding validation loss
print(f"Best Batch Size: {best_batch_size}, Best Dropout Rate: {best_dropout_rate}, Best Validation Loss: {best_validation_loss}")