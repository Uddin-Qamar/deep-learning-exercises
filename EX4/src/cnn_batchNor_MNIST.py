# CNN implementation on MNIST Dataset with and without Batch normalization to see the diffrence
# Initialy this code was run on the google colab



import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Reproducibility

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA deterministic (may slow down a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Models
class CNN_NoBN(nn.Module):
    """
    Simple CNN for MNIST without BatchNorm.
    Input: (N, 1, 28, 28)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # -> (N, 32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # -> (N, 64, 14, 14) after pool
        self.pool = nn.MaxPool2d(2, 2)                # halves H,W each time

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (N, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (N, 64, 7, 7)
        x = x.view(x.size(0), -1)             # (N, 64*7*7)
        x = F.relu(self.fc1(x))               # (N, 128)
        x = self.fc2(x)                       # (N, 10) logits
        return x


class CNN_BN(nn.Module):
    """
    Same CNN but with BatchNorm.
    Common pattern: Conv -> BN -> ReLU -> Pool
                    Linear -> BN -> ReLU
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x


# Train / Eval helpers

@torch.no_grad()
def classification_error_percent(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    acc = correct / total
    return 100.0 * (1.0 - acc)


def train_one_model(model, train_loader, test_loader, device, epochs=10, lr=1e-3):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_errs = []
    test_errs = []

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        train_err = classification_error_percent(model, train_loader, device)
        test_err = classification_error_percent(model, test_loader, device)
        train_errs.append(train_err)
        test_errs.append(test_err)

        print(f"Epoch {epoch:02d} | Train error: {train_err:6.2f}% | Test error: {test_err:6.2f}%")

    return train_errs, test_errs


# Main experiment

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Dataset normalization (input normalization)
    # Note: this is NOT BatchNorm; BatchNorm is inside the model.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    epochs = 10
    lr = 1e-3

    print("\n=== Training WITHOUT BatchNorm ===")
    no_bn_model = CNN_NoBN()
    no_bn_train_err, no_bn_test_err = train_one_model(
        no_bn_model, train_loader, test_loader, device, epochs=epochs, lr=lr
    )

    print("\n=== Training WITH BatchNorm ===")
    bn_model = CNN_BN()
    bn_train_err, bn_test_err = train_one_model(
        bn_model, train_loader, test_loader, device, epochs=epochs, lr=lr
    )

    # Display classification error (%) by epoch (test)
    print("\n=== Classification error (%) by epoch (TEST) ===")
    print("Epoch | No BN  | With BN")
    print("------------------------")
    for i in range(epochs):
        print(f"{i+1:5d} | {no_bn_test_err[i]:6.2f}% | {bn_test_err[i]:7.2f}%")

    # Plot
    plt.figure()
    plt.plot(range(1, epochs + 1), no_bn_test_err, label="No BatchNorm (Test error %)")
    plt.plot(range(1, epochs + 1), bn_test_err, label="With BatchNorm (Test error %)")
    plt.xlabel("Epoch")
    plt.ylabel("Classification error (%)")
    plt.title("MNIST: CNN with vs without Batch Normalization")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()