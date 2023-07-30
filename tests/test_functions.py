import os
import glob
import pytest

from functions import make_val_dataset

@pytest.fixture
def setup_test_environment(tmpdir):
    # Create temporary training and validation directories
    training_folder = tmpdir.mkdir("Training")
    val_folder = tmpdir.mkdir("Validation")

    # Create sample image files in the temporary training directory
    num_image_files = 20
    for i in range(num_image_files):
        class_folder = os.path.join(training_folder, f"Class_{i // 5}")
        os.makedirs(class_folder, exist_ok=True)
        image_file = os.path.join(class_folder, f"image_{i}.jpg")
        with open(image_file, "w"):
            pass

    return str(training_folder), str(val_folder)

def test_make_val_dataset(setup_test_environment):
    # Get the paths of the temporary training and validation folders
    training_folder, val_folder = setup_test_environment

    # Specify the validation split percentage for testing
    val_split = 0.2

    # Call the function to be tested
    make_val_dataset(training_folder, val_split)

    # Check if the images are moved correctly
    for class_folder in glob.glob(os.path.join(val_folder, "Class_*")):
        class_files = os.listdir(class_folder)
        assert len(class_files) == 4  # 20% of 20 image files is 4

    # Check if the images are removed from the training folder
    training_files = glob.glob(os.path.join(training_folder, "**/*.jpg"), recursive=True)
    assert len(training_files) == 16  # 20 - 4 = 16