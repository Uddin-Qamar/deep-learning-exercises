# CNN model was traind on MNIST dataset for data augmentation and tested with the single handwritten digit 
# The model correctly classify the number 
# The code was initial run and tested on the google colab

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# MNIST normalization values
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=256, shuffle=False)

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),               # 28x28 → 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)                # 14x14 → 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MNIST_CNN().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch():
    model.train()
    total_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(trainloader)

epochs = 5
for epoch in range(epochs):
    loss = train_one_epoch()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

@torch.no_grad()
def evaluate():
    model.eval()
    correct, total = 0, 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    print("Test accuracy:", correct / total)

evaluate()

from google.colab import files
uploaded = files.upload()

img = Image.open("handwritten6.jpg").convert("L")  # grayscale
img = img.resize((28, 28))

img_tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])

img_tensor = img_tf(img).unsqueeze(0).to(device)  # (1,1,28,28)

model.eval()
with torch.no_grad():
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(1).item()
    confidence = probs[0, pred].item()

print(f"Predicted digit: {pred}")
print(f"Confidence: {confidence:.4f}")

plt.imshow(img, cmap="gray")
plt.title(f"Predicted digit: {pred}")
plt.axis("off")
plt.show()