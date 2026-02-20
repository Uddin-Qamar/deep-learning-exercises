# Custom CNN model on MNIST dataset with following 5 activation functions including : Sigmoid, Tanh, ReLU, ELU, SELU
# Using pytorch lib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available(), torch.cuda.get_device_name(0)
print("Using device:", device)

batch_size = 40
epochs = 5   # increase if you want better performance
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
])

# Download + load train data
full_train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Split train into train + validation
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 4. CNN Model that Accepts Any Activation Function
class SimpleCNN(nn.Module):
    def __init__(self, activation_layer):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            activation_layer,
            nn.MaxPool2d(2, 2),   # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            activation_layer,
            nn.MaxPool2d(2, 2)    # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            activation_layer,
            nn.Linear(128, 10)  # 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#Training & Evaluation Functions
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

# Train with Different Activation Functions
activation_dict = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "selu": nn.SELU()
}

histories = {}  # to store loss/acc for each activation




for act_name, act_layer in activation_dict.items():
    print(f"\n=== Training with activation: {act_name} ===")
    model = SimpleCNN(activation_layer=act_layer).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses   = []
    train_accs   = []
    val_accs     = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Evaluate on test set at the end
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy with {act_name}: {test_acc:.4f}")

    histories[act_name] = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "test_acc": test_acc
    }

#Plot: Loss & Accuracy per Epoch (for each activation)
epochs_range = range(1, epochs + 1)

# Plot loss
plt.figure(figsize=(10, 5))
for act_name, history in histories.items():
    plt.plot(epochs_range, history["train_loss"], linestyle='-', label=f'{act_name} train')
    plt.plot(epochs_range, history["val_loss"], linestyle='--', label=f'{act_name} val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss for Different Activations')
plt.legend()
plt.grid(True)
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 5))
for act_name, history in histories.items():
    plt.plot(epochs_range, history["train_acc"], linestyle='-', label=f'{act_name} train')
    plt.plot(epochs_range, history["val_acc"], linestyle='--', label=f'{act_name} val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy for Different Activations')
plt.legend()
plt.grid(True)
plt.show()

# Which activation is most suitable for MNIST?



print("\n==============================")
print("In my Trainig and testing of CNN model, both RELU and ELU perform well and have same accoracy value in testing for MNIST Dataset")
print("\n==============================")

best_val_acc = 0
best_acts = []   # list to store all best activation functions

for act_name, history in histories.items():
    final_val_acc = history['val_acc'][-1]  # only final value
    test_acc = history['test_acc']

    print(f"{act_name}: final val acc = {final_val_acc:.4f}, "
          f"test acc = {test_acc:.4f}")

    # compare only FINAL val accuracy
    if test_acc > best_val_acc:
        best_val_acc = test_acc
        best_acts = [act_name]

    elif test_acc == best_val_acc:
        best_acts.append(act_name)

print("\n==============================")
print(f"🔥 Best activation function(s): {', '.join(best_acts)}")
print(f"🔥 Best final validation accuracy: {best_val_acc:.4f}")