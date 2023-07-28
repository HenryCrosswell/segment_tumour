import torch
import torch.nn as nn
from tqdm import tqdm

# Define the U-Net architecture for brain tumor segmentation with grayscale images
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Linear(64 * 112 * 112, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training", ncols=80)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix({"Loss": total_loss / (len(train_loader) * train_loader.batch_size)})

        print(f"Epoch {epoch + 1} - Training Loss: {total_loss / len(train_loader)}")

        # Validation phase
        model.eval()
        total_val_loss = 0.0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation", ncols=80)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
                val_bar.set_postfix({"Loss": total_val_loss / (len(val_loader) * val_loader.batch_size)})

        print(f"Epoch {epoch + 1} - Validation Loss: {total_val_loss / len(val_loader)}")


def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class indices
            total_correct += (predicted == labels.squeeze()).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy * 100:.2f}%")