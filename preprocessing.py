import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


class BrainTumorDataset(Dataset):
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

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label