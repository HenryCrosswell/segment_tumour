import os
import pytest
import shutil
from functions import BrainTumourDataset, make_val_dataset
from pathlib import Path
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation


def test_BrainTumourDataset():

    current_working_directory = Path.cwd()
    test_data_folder = current_working_directory.joinpath('tests/test_data')

    dataset = BrainTumourDataset(test_data_folder)

    assert len(dataset) == 8  # 4 folders with 5 images each

    image, label = dataset[0]
    assert image.size == (512, 512) 
    assert image.mode == 'L'  # The image should be converted to greyscale
    assert label == 0

def test_make_val_dataset(test_data_folder):
    data_folder = test_data_folder
    training_folder = os.path.join(data_folder, 'Training')
    validation_folder = os.path.join(data_folder, 'Validation')

    # Create a dummy Training folder
    os.makedirs(training_folder, exist_ok=True)
    for folder in ['glioma', 'meningioma', 'notumor', 'pituitary']:
        folder_path = os.path.join(training_folder, folder)
        os.makedirs(folder_path, exist_ok=True)

    # Call the function to create the validation dataset
    make_val_dataset(training_folder, 'glioma', 0.2)

    # Assert that the validation folder and its content are created
    assert os.path.exists(os.path.join(validation_folder, 'glioma'))
    assert len(os.listdir(os.path.join(validation_folder, 'glioma'))) == 1

    # Clean up
    shutil.rmtree(training_folder)
