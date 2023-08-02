import os
import shutil
from functions import BrainTumourDataset, create_validation_dataset
from pathlib import Path
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation


def test_BrainTumourDataset():

    current_working_directory = Path.cwd()
    test_data_folder = current_working_directory.joinpath(Path('tests/test_data/Training'))

    dataset = BrainTumourDataset(test_data_folder)

    assert len(dataset) == 8  # 4 folders with 5 images each

    image, label = dataset[0]
    assert image.size == (512, 512) 
    assert image.mode == 'L'  # The image should be converted to greyscale
    assert label == 0

def test_create_validation_dataset():
    current_working_directory = Path.cwd()
    test_data_folder = current_working_directory.joinpath(Path('tests/test_data/'))
    training_folder = os.path.normpath(os.path.join(test_data_folder, 'Training'))
    validation_folder = os.path.normpath(os.path.join(test_data_folder, 'Validation'))

    # Call the function to create the validation dataset
    create_validation_dataset(training_folder, 0.2)

    # Assert that the validation folder and its content are created
    assert os.path.exists(os.path.join(validation_folder, 'glioma'))
    assert len(os.listdir(os.path.join(validation_folder, 'glioma'))) == 1

    # Clean up
    shutil.rmtree(validation_folder)
