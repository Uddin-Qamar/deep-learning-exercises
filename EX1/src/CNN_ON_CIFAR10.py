
# Custom CNN model on CIFAR-10 with following 5 activation functions including : Sigmoid, Tanh, ReLU, ELU, SELU
# Using pytorch lib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -------------------------
# 1. Device & hyperparams
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

batch_size = 128
num_epochs = 10
lr = 1e-3 # i.e lr = 0.001

# -------------------------
# 2. CIFAR-10 data
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

train_ds = datasets.CIFAR10(root="./data", train=True,
                            download=True, transform=transform)
test_ds = datasets.CIFAR10(root="./data", train=False,
                           download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True)
val_loader = DataLoader(test_ds, batch_size=batch_size,
                        shuffle=False)

# -------------------------
# 3. Helper: choose activation
# -------------------------
def get_activation(name):
    name = name.lower()
    if name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "selu":
        return nn.SELU()
    else:
        raise ValueError("Unknown activation:", name)

# -------------------------
# 4. CNN model
# -------------------------
class QamarCIFAR10CNN(nn.Module):
    def __init__(self, act_name="relu"):
        super().__init__()
        self.act = get_activation(act_name)

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 32->16->8

        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 10)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(self.act(self.fc1(x)))
        x = self.fc2(x)  # logits
        return x

# -------------------------
# 5. Train function (1 activation)
# -------------------------
def train_model(act_name):
    print(f"\n=== Activation: {act_name} ===")
    model = QamarCIFAR10CNN(act_name).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Validate ----
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = crit(out, y)
                running_loss += loss.item() * x.size(0)
                _, pred = out.max(1)
                total += y.size(0)
                correct += (pred == y).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, "
              f"Train Acc: {train_acc*100:.1f}%, Val Acc: {val_acc*100:.1f}%")

    return history

# -------------------------
# 6. Run for all activations
# -------------------------
activations = ["sigmoid", "tanh", "relu", "elu", "selu"]
all_hist = {}

for act in activations:
    all_hist[act] = train_model(act)

# -------------------------
# 7. Plot curves
# -------------------------
def plot_metric(metric_name, title, ylabel):
    plt.figure(figsize=(7, 5))
    for act in activations:
        plt.plot(all_hist[act][metric_name], label=act)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Training & validation loss
plot_metric("train_loss", "Training Loss vs Epoch", "Loss")
plot_metric("val_loss", "Validation Loss vs Epoch", "Loss")

# Training & validation accuracy
plot_metric("train_acc", "Training Accuracy vs Epoch", "Accuracy")
plot_metric("val_acc", "Validation Accuracy vs Epoch", "Accuracy")

# -------------------------
# 8. Print best val accuracy
# -------------------------
print("\nBest validation accuracy for each activation:")
for act in activations:
    best_val = max(all_hist[act]["val_acc"]) * 100
    print(f"{act:7s}: {best_val:.2f}%")