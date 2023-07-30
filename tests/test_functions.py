import os
import pytest
import shutil
from segment_tumour import BrainTumourDataset, make_val_dataset

@pytest.fixture
def sample_data_folder(tmpdir):
    data_folder = tmpdir.mkdir("data")
    image_folders = ['glioma', 'meningioma', 'notumor', 'pituitary']
    for folder in image_folders:
        folder_path = data_folder.mkdir(folder)
        for i in range(5):
            # Create dummy image files for testing
            open(os.path.join(folder_path, f'image_{i}.jpg'), 'a').close()
    return str(data_folder)

def test_BrainTumourDataset(sample_data_folder):
    dataset = BrainTumourDataset(sample_data_folder)
    assert len(dataset) == 20  # 4 folders with 5 images each

    image, label = dataset[0]
    assert image.size == (224, 224)  # The image should be resized to (224, 224)
    assert image.mode == 'L'  # The image should be converted to greyscale

def test_make_val_dataset(sample_data_folder):
    data_folder = sample_data_folder
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
