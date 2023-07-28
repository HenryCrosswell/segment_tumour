import torch
import pytest
from preprocessing import BrainTumorDataset
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize

# Define a sample data folder for testing
sample_data_folder = "/Users/henrycrosswell/Downloads/Coding Project/tumour_segment/segment_tumour/tests/test_data"
sample_transform = Compose([
    Resize((224, 224)),                # Resize the image to a fixed size
    Grayscale(num_output_channels=1),  # Convert the image to grayscale
    ToTensor(),                        # Convert the image to a PyTorch tensor
    Normalize(mean=[0.5], std=[0.5]),  # Normalize the pixel values
])

# Use the BrainTumorDataset for testing
@pytest.fixture
def brain_tumor_dataset():
    return BrainTumorDataset(data_folder=sample_data_folder, transform=sample_transform)

def test_dataset_sample(brain_tumor_dataset):
    # Check individual samples in the dataset
    for index in range(len(brain_tumor_dataset)):
        image, label = brain_tumor_dataset[index]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert image.shape[0] == 1  # Grayscale images, so they have a single channel
        assert image.shape[1:] == (224, 224)  # Image size is (224, 224)
        assert label.shape == (1, 1)  # Labels are 1x1 tensors

def test_dataset_length(brain_tumor_dataset):
    # Check the length of the dataset
    assert len(brain_tumor_dataset) == 8  # Two sample images in the dataset


def test_dataset_labels(brain_tumor_dataset):
    # Check the labels for individual samples in the dataset
    assert brain_tumor_dataset[0][1] == 1
    assert brain_tumor_dataset[1][1] == 2 
    assert brain_tumor_dataset[2][1] == 3
    assert brain_tumor_dataset[3][1] == 4
    

def test_dataset_invalid_index(brain_tumor_dataset):
    # Check behavior for an invalid index (out of range)
    with pytest.raises(IndexError):
        brain_tumor_dataset[len(brain_tumor_dataset)]


def test_dataset_transform_applied(brain_tumor_dataset):
    # Check if the transform is applied to the images
    transform = Compose([
        Resize((224, 224)),
        Grayscale(num_output_channels=1),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5]),
    ])

    # Get the first image before the transform
    original_image, _ = BrainTumorDataset(data_folder=sample_data_folder, transform=None)[0]

    # Apply the transform to the original image
    transformed_original_image = transform(original_image)

    # Get the first image after applying the transform using the dataset
    transformed_image, _ = brain_tumor_dataset[0]

    # Check if the transformed image matches the expected transform result
    assert torch.all(torch.eq(transformed_original_image, transformed_image))

def test_dataset_label_shape(brain_tumor_dataset):
    # Check the shape of the labels for individual samples in the dataset
    for index in range(len(brain_tumor_dataset)):
        _, label = brain_tumor_dataset[index]
        assert label.shape == (1, 1)  # Labels are expected to have shape (1, 1)