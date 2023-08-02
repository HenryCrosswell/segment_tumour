import os
import glob
import random 
import shutil
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from torch.utils.data import Dataset
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

def create_validation_dataset(archive_path, validation_ratio=0.2):
    """
    Create a validation dataset from the training dataset.

    Args:
        archive_path (str): Path to the main archive folder containing 'Training' and 'Testing' folders.
        validation_ratio (float, optional): Proportion of data to use for validation (default is 0.2).

    Returns:
        None
    """
    training_folder = os.path.join(archive_path, 'Training')
    validation_folder = os.path.join(archive_path, 'Validation')

    # Create the validation folder if it doesn't exist
    os.makedirs(validation_folder, exist_ok=True)

    # Iterate over each subfolder in the training folder and create the corresponding validation subfolder
    for folder in ['glioma', 'meningioma', 'notumor', 'pituitary']:
        train_image_folder = os.path.join(training_folder, folder)
        val_image_folder = os.path.join(validation_folder, folder)

        # Create the validation subfolder if it doesn't exist
        os.makedirs(val_image_folder, exist_ok=True)

        # Get a list of image files in the training folder
        image_files = os.listdir(train_image_folder)

        # Calculate the number of images to move to the validation folder
        num_val_images = int(len(image_files) * validation_ratio)

        # Randomly select images to move
        selected_images = random.sample(image_files, num_val_images)

        # Move the selected images to the validation folder
        for image_file in selected_images:
            train_image_path = os.path.join(train_image_folder, image_file)
            val_image_path = os.path.join(val_image_folder, image_file)
            shutil.move(train_image_path, val_image_path)

    print("Validation dataset created successfully.")
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
    
