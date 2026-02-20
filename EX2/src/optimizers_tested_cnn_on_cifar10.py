
# CIFAR-10 + CNN + SGD tuning (learning rate & momentum)
# Baseline experiment for optimizer hyperparameters
# This code was run on google colab 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -------------------------
# 1. Device & global hyperparams
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 128
num_epochs = 5

# -------------------------
# 2. CIFAR-10 dataset & loaders
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
# 3. CNN model ReLU only
# -------------------------
class QamarCIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

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
# 4. Train function for a single (lr, momentum)
# -------------------------
def make_optimizer(optimizer_name, model, lr, momentum=0.9):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "rmsprop":
        # RMSProp also has a momentum parameter
        return optim.RMSprop(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "adagrad":
        # Adagrad doesn't usually use momentum
        return optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == "adam":
        # Adam has betas instead of momentum; we keep defaults
        return optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_model(optimizer_name, lr, momentum=0.9):
    print(f"\n=== {optimizer_name.upper()} (lr={lr}, momentum={momentum if optimizer_name in ['sgd','rmsprop'] else 'N/A'}) ===")
    model = QamarCIFAR10CNN().to(device)
    crit = nn.CrossEntropyLoss()
    opt = make_optimizer(optimizer_name, model, lr, momentum)

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
# 5. Hyperparameter grid: learning rate & momentum
# -------------------------
optimizers = ["sgd", "rmsprop", "adagrad", "adam"]
lrs = [0.1, 0.01, 0.001]

results = {}       # (opt_name, lr) -> history
best_for_opt = {}  # opt_name -> (best_lr, best_val_acc, history)

for opt_name in optimizers:
    best_lr = None
    best_acc = 0.0
    best_hist = None

    for lr in lrs:
        # For SGD & RMSProp: use momentum; for others, it's ignored
        hist = train_model(opt_name, lr=lr, momentum=0.9)
        results[(opt_name, lr)] = hist

        max_val_acc = max(hist["val_acc"])
        print(f"--> {opt_name.upper()} lr={lr}: best val acc={max_val_acc*100:.2f}%")

        if max_val_acc > best_acc:
            best_acc = max_val_acc
            best_lr = lr
            best_hist = hist

    best_for_opt[opt_name] = (best_lr, best_acc, best_hist)

print("\nSummary of best validation accuracy per optimizer:")
for opt_name, (lr, acc, _) in best_for_opt.items():
    print(f"{opt_name.upper():7s}: best lr={lr}, best val acc={acc*100:.2f}%")

# -------------------------
# 6. Plot curves for the best configuration
# -------------------------
# Plot validation accuracy for each optimizer's best setting
plt.figure(figsize=(7,5))
for opt_name, (lr, acc, hist) in best_for_opt.items():
    plt.plot(hist["val_acc"], label=f"{opt_name.upper()} (lr={lr})")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Best validation accuracy per optimizer")
plt.legend()
plt.grid(True)
plt.show()

from google.colab import files
plt.savefig("my_plot.png")
files.download("my_plot.png")

