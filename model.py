import random 
import torch
import torch.nn as nn
from tqdm import tqdm
import logging

class CNNModel(nn.Module):
    """
    CNNModel: Convolutional Neural Network model for brain tumour segmentation.

    Args:
        num_classes (int): Number of output classes (tumour classes).
        dropout_prob (float): Dropout probability to apply to dropout layers.
    """
    def __init__(self, num_classes, dropout_rate):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Add dropout after the fourth convolutional layer
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class TumourSegmentationTrainer:
    """
    TumourSegmentationTrainer: Trainer class for training and testing the tumour segmentation model.

    Args:
        model (torch.nn.Module): The CNN model for tumour segmentation.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        criterion (torch.nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        device (str): Device to use for training ('cuda' or 'cpu').
    """
    def __init__(self, model=None, train_loader=None, val_loader=None, test_loader=None, criterion=None, optimizer=None, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # Move the model to the appropriate device
        if self.model and self.device:
            self.model.to(device)

    def random_hyperparameter_search(self, train_loader, val_loader, test_loader, num_epochs=10, num_iterations=10):
        """
        Perform random hyperparameter search.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            test_loader (torch.utils.data.DataLoader): DataLoader for test data.
            num_epochs (int, optional): Number of training epochs for each hyperparameter combination. Default is 10.
            num_iterations (int, optional): Number of random hyperparameter combinations to try. Default is 10.

        Returns:
            dict: Dictionary containing the best hyperparameters and corresponding test accuracy.
        """
        best_accuracy = 0.0
        best_hyperparameters = {}

        for _ in range(num_iterations):
            # Sample random hyperparameters from the search space
            batch_size = random.choice([16, 32, 64])
            dropout_rate = random.choice([0.2, 0.4, 0.6])
            learning_rate = random.uniform(0.0001, 0.01)

            # Initialize the model with the random hyperparameters
            model = CNNModel(num_classes=4, dropout_rate=dropout_rate)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Initialize the trainer with the current hyperparameters
            trainer = TumourSegmentationTrainer(model, train_loader, val_loader, test_loader, criterion, optimizer, device)

            # Train the model
            trainer.train(num_epochs=num_epochs)

            # Evaluate the model on the test data
            accuracy = trainer.test(test_loader)

            # Update the best hyperparameters if the current hyperparameter combination performs better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyperparameters = {
                    "batch_size": batch_size,
                    "dropout_rate": dropout_rate,
                    "learning_rate": learning_rate,
                    "test_accuracy": accuracy
                }

        return best_hyperparameters
    
    def train(self, num_epochs=10, log_interval=10):
        """
        Train the tumour segmentation model.

        Args:
            num_epochs (int, optional): Number of training epochs. Default is 10.
            log_interval (int, optional): Interval for logging the training progress. Default is 10.
        """
        logging.info("Start training...")
        best_val_loss = float('inf')
        patience = 5
        early_stop_counter = 0

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            total_loss = 0.0

            train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} - Training", ncols=80)
            for batch_idx, (inputs, labels) in enumerate(train_bar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if batch_idx % log_interval == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    train_bar.set_postfix({"Training Loss": avg_loss})

            avg_loss = total_loss / len(self.train_loader)
            logging.info(f"Epoch {epoch + 1} - Training Loss: {avg_loss}")

            # Validation phase (moved outside the training loop)
            val_loss = self.evaluate_with_progress(self.val_loader, phase= 'Validation')
            logging.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logging.info("Early stopping triggered.")
                    break

    def evaluate_with_progress(self, data_loader, phase):
        """
        Evaluate the model on a given data loader with progress bar.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
            phase (str): Phase for the progress bar (e.g., "Validation" or "Testing").

        Returns:
            float: Average evaluation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            eval_bar = tqdm(data_loader, desc=phase, ncols=80)
            for inputs, labels in eval_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                avg_loss = total_loss / len(eval_bar)
                eval_bar.set_postfix({f"{phase} Loss": avg_loss})

        avg_loss = total_loss / len(data_loader)
        return avg_loss

def test_model(trained_model, test_loader, device):
    """
    Evaluate the trained model on the test dataset.

    Args:
        trained_model (torch.nn.Module): Trained model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.

    Returns:
        float: Test accuracy.
    """
    trained_model.eval()
    total_correct = 0
    total_samples = 0

    test_bar = tqdm(test_loader, desc="Testing", ncols=80)
    with torch.no_grad():
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class indices
            total_correct += (predicted == labels.squeeze()).sum().item()
            total_samples += labels.size(0)

            # Update the progress bar with the current test accuracy
            test_accuracy = total_correct / total_samples
            test_bar.set_postfix({"Test Accuracy": f"{test_accuracy * 100:.2f}%"})

    accuracy = total_correct / total_samples
    logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy