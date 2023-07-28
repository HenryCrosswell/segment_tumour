import os
import glob
import random 
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from tqdm import tqdm
import logging
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

class BrainTumourDataset(Dataset):
    """
    BrainTumourDataset: Custom dataset for loading brain tumour images and labels from folder name.

    Args:
        data_folder (str): Path to the folder containing the images.
        transform  (optional): Data transformations to apply to the images.
    """
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.image_folders = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.images = []
        self.labels = []

        for label, folder in enumerate(self.image_folders):
            image_folder = os.path.join(data_folder, folder)
            image_files = os.listdir(image_folder)
            self.images.extend([os.path.join(image_folder, file) for file in image_files])
            self.labels.extend([label] * len(image_files))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]

        # Convert to greyscale images
        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label

def make_val_dataset(training_folder, data_folder, val_split):
    """
    Create a validation dataset from the training dataset.

    Args:
        training_folder (str): Path to the folder containing the training data.
        data_folder (str): Name of the data folder (class name).
        val_split (float): Proportion of data to use for validation.

    Returns:
        None
    """
    # Get image files
    image_files = glob.glob(os.path.join(training_folder, data_folder, '*.jpg'))
    num_val_images = int(len(image_files) * val_split)

    # Create the validation directory
    val_dir = os.path.join(training_folder, 'Validation', data_folder)
    os.makedirs(val_dir, exist_ok=True)

    # Move the selected images to the validation directory
    selected_images = random.sample(image_files, num_val_images)
    for image_path in selected_images:
        val_image_path = os.path.join(val_dir, os.path.basename(image_path))
        shutil.move(image_path, val_image_path)

# Define the U-Net architecture for brain tumour segmentation with grayscale images
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

def get_data_loaders(train_folder, validation_folder, test_data_folder, batch_size=32):
    """
    get_data_loaders: Load and prepare data loaders for training, validation, and testing.

    Args:
        train_folder (str): Path to the folder containing training data.
        validation_folder (str): Path to the folder containing validation data.
        test_data_folder (str): Path to the folder containing test data.
        batch_size (int, optional): Batch size for data loaders. Default is 32.

    Returns:
        train_loader : DataLoader for training data.
        val_loader : DataLoader for validation data.
        test_loader : DataLoader for test data.
    """
    transform = Compose([
        Resize((224, 224)),                  # standardises image size
        RandomHorizontalFlip(),              # randomly flips and rotates image
        RandomVerticalFlip(),                # "
        RandomRotation(15),                  # "
        ToTensor(),                          # converts image to pytorch tensor
        Normalize(mean=[0.5], std=[0.5]),    # helps normalise pixel values to aid training
    ])

    train_dataset = BrainTumourDataset(train_folder, transform=transform)
    val_dataset = BrainTumourDataset(validation_folder, transform=transform)
    test_dataset = BrainTumourDataset(test_data_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

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
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # Move the model to the appropriate device
        self.model.to(device)

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

            avg_loss = total_loss / len(self.train_loader)
            logging.info(f"Epoch {epoch + 1} - Training Loss: {avg_loss}")

            # Validation phase
            val_loss = self.evaluate(self.val_loader)
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

        # Save the trained model
        torch.save(self.model.state_dict(), 'model.pth')
        logging.info("Model saved.")

    def evaluate(self, data_loader):
        """
        Evaluate the model on a given data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.

        Returns:
            float: Average evaluation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    def test(self, test_loader):
        """
        Evaluate the tumour segmentation model on the test dataset.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for test data.

        Returns:
            float: Test accuracy.
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)  # Get the predicted class indices
                total_correct += (predicted == labels.squeeze()).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy