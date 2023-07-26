from functions import make_val_dataset
from preprocessing import BrainTumorDataset
from model import tumour_unet
import os
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from pathlib import Path

training_path = Path('F:/Users\Henry/Coding/Tumour Segmentation/archive/Training')
data_folder= Path('F:/Users/Henry/Coding/Tumour Segmentation/archive/')

make_val_dataset(training_path, data_folder, 0.2)

transform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

train_dataset = BrainTumorDataset(os.path.join(data_folder, 'Training'), transform=transform)
val_dataset = BrainTumorDataset(os.path.join(data_folder, 'val'), transform=transform)
test_dataset = BrainTumorDataset(os.path.join(data_folder, 'Testing'), transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)