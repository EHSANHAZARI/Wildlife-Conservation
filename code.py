# ============================================================
# Imports
# ============================================================
import os
from collections import Counter

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from training import train, predict  # custom training utilities

# ============================================================
# Setup
# ============================================================
torch.backends.cudnn.deterministic = True  # for reproducibility

# Select device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ============================================================
# Dataset setup
# ============================================================
train_dir = "sea_creatures/train"
classes = os.listdir(train_dir)
print("Classes:", classes)

height, width = 224, 224

# Custom transform to ensure all images are converted to RGB
class ConvertToRGB:
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

# ============================================================
# Transform pipeline
# ============================================================
transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((height, width)),
    transforms.ToTensor()
])
print("Transform:", transform)

# ============================================================
# Create dataset
# ============================================================
dataset = datasets.ImageFolder(root=train_dir, transform=transform)
print("Sample image size:", dataset[0][0].shape)
print("Sample label:", dataset[0][1])

# ============================================================
# Class distribution
# ============================================================
counts = Counter(x[1] for x in tqdm(dataset))
print("Counts (index-based):", counts)
print("Class-to-idx mapping:", dataset.class_to_idx)

class_distribution = {
    class_name: counts[idx] for class_name, idx in dataset.class_to_idx.items()
}
print("Class distribution:", class_distribution)

# ============================================================
# DataLoader
# ============================================================
batch_size = 32
dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

first_batch = next(iter(dataset_loader))
print(f"Shape of one batch: {first_batch[0].shape}")
print(f"Shape of labels: {first_batch[1].shape}")

# ============================================================
# Compute mean & std for normalization
# ============================================================
def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

mean, std = get_mean_std(DataLoader(dataset, batch_size=1, shuffle=False))
print("Mean:", mean)
print("Std:", std)

# ============================================================
# Normalized transform & dataset
# ============================================================
transform_norm = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])
print("Normalized transform:", transform_norm)

norm_dataset = datasets.ImageFolder(root=train_dir, transform=transform_norm)
print("Sample normalized image size:", norm_dataset[0][0].shape)
print("Sample normalized label:", norm_dataset[0][1])

# ============================================================
# Train/Validation split
# ============================================================
train_size = int(0.8 * len(norm_dataset))
val_size = len(norm_dataset) - train_size
g = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    norm_dataset, [train_size, val_size], generator=g
)

print("Training size:", len(train_dataset))
print("Validation size:", len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ============================================================
# Define CNN model
# ============================================================
torch.manual_seed(42)
model = nn.Sequential()

# Convolutional Block 1
model.append(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1))
model.append(nn.ReLU())
model.append(nn.MaxPool2d(kernel_size=4, stride=4))

# Convolutional Block 2
model.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1))
model.append(nn.ReLU())
model.append(nn.MaxPool2d(kernel_size=4, stride=4))

# Convolutional Block 3
model.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
model.append(nn.ReLU())
model.append(nn.MaxPool2d(kernel_size=4, stride=4))
model.append(nn.Flatten())

# Fully Connected Layers
model.append(nn.Dropout(p=0.5))
model.append(nn.Linear(in_features=576, out_features=500))
model.append(nn.ReLU())
model.append(nn.Dropout(p=0.5))
model.append(nn.Linear(in_features=500, out_features=len(classes)))

# Print model summary
summary(model, input_size=(batch_size, 3, height, width))

# ============================================================
# Loss & optimizer
# ============================================================
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Send model to device
model = model.to(device)

# ============================================================
# Training
# ============================================================
epochs = 10
train(model, optimizer, loss_fn, train_loader, val_loader, epochs=epochs, device=device)

# ============================================================
# Predictions on validation set
# ============================================================
probabilities = predict(model, val_loader, device=device)
predictions = torch.argmax(probabilities, dim=1)
print("Number of validation predictions:", predictions.shape)

# ============================================================
# Confusion matrix
# ============================================================
targets = []
for _, labels in tqdm(val_loader):
    targets.extend(labels.tolist())

fig, ax = plt.subplots(figsize=(10, 6))
cm = confusion_matrix(targets, predictions.cpu().numpy())
classes = list(dataset.class_to_idx.keys())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical", ax=ax)
plt.show()

# ============================================================
# Test dataset & loader
# ============================================================
test_dir = "sea_creatures/test"
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_norm)
print("Number of test images:", len(test_dataset))

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============================================================
# Test predictions
# ============================================================
test_probabilities = predict(model, test_loader, device=device)
test_predictions = torch.argmax(test_probabilities, dim=1)
print("Number of test predictions:", test_predictions.shape)
