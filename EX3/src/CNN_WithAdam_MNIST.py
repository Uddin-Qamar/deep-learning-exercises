# Implementation of custom CNN model with Adam node function on MNIST dataset 
# The program was run on google colab


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -----------------------
# 1. Hyperparameters
# -----------------------
batch_size = 64
num_epochs = 10
learning_rate = 1e-3      # for Adam
weight_decay = 1e-4       # L2 regularization

# -----------------------
# 2. Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# 3. Data loading (MNIST)
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
])

# Download + load dataset
full_train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

# Split training into train/validation
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# -----------------------
# 4. CNN Model
#    "Node function" = ReLU activation
# -----------------------
class CNNForMNIST(nn.Module):
    def __init__(self):
        super(CNNForMNIST, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1x28x28 -> 32x28x28
            nn.ReLU(),                                   # node function
            nn.MaxPool2d(2),                             # 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)                              # 64x7x7
        )
        self.fc_block = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)    # 10 classes
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_block(x)
        return x

model = CNNForMNIST().to(device)

# -----------------------
# 5. Loss and Optimizer (Adam)
# -----------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(0.9, 0.999),  # default Adam betas
    eps=1e-8
)

# -----------------------
# 6. Helper: compute accuracy
# -----------------------
def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total

# -----------------------
# 7. Training & Validation loops
# -----------------------
train_losses = []
val_losses   = []
train_accs   = []
val_accs     = []

best_val_acc = 0.0
best_epoch   = 0

for epoch in range(1, num_epochs + 1):
    # ---- Training ----
    model.train()
    epoch_train_loss = 0.0
    epoch_train_acc = 0.0
    num_train_batches = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        epoch_train_acc += compute_accuracy(outputs, labels)
        num_train_batches += 1

    epoch_train_loss /= num_train_batches
    epoch_train_acc  /= num_train_batches

    # ---- Validation ----
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_acc = 0.0
    num_val_batches = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            epoch_val_loss += loss.item()
            epoch_val_acc += compute_accuracy(outputs, labels)
            num_val_batches += 1

    epoch_val_loss /= num_val_batches
    epoch_val_acc  /= num_val_batches

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accs.append(epoch_train_acc)
    val_accs.append(epoch_val_acc)

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        best_epoch = epoch

    print(
        f"Epoch [{epoch}/{num_epochs}] "
        f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
        f"Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}"
    )

print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

# -----------------------
# 8. Test set evaluation (optional)
# -----------------------
model.eval()
test_acc = 0.0
num_test_batches = 0

with torch.inference_mode():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_acc += compute_accuracy(outputs, labels)
        num_test_batches += 1

test_acc /= num_test_batches
print(f"Test Accuracy: {test_acc:.4f}")

# -----------------------
# 9. Plotting results
# -----------------------
epochs = range(1, num_epochs + 1)

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(epochs, train_accs, label="Train Accuracy")
plt.plot(epochs, val_accs, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()