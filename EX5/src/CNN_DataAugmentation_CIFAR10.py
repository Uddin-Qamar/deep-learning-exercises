# CNN model was traind on MNIST dataset for data augmentation and tested with the single dog image 
# The model correctly classify the dog
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

# CIFAR-10 class names
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Standard CIFAR-10 normalization (commonly used)
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)

train_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
])

test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
testloader  = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

class MyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 32x32 -> 16x16
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 16x16 -> 8x8
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MyCNN().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return correct/total

epochs = 10  # increase to 20 for better accuracy if you want
for ep in range(1, epochs+1):
    tr_loss, tr_acc = train_one_epoch(model, trainloader)
    te_acc = evaluate(model, testloader)
    print(f"Epoch {ep:02d}: train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f}, test_acc={te_acc:.4f}")

from google.colab import files
uploaded = files.upload()

img_tf = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(mean, std)
])

img = Image.open("dog.jpg").convert("RGB")

img_orig = img
img_hflip = T.functional.hflip(img)   # horizontal flip (left-right)
img_vflip = T.functional.vflip(img)   # vertical flip (up-down)

@torch.no_grad()
def predict_image(pil_img):
    model.eval()
    x = img_tf(pil_img).unsqueeze(0).to(device)  # (1,3,32,32)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)  # (10,)
    pred_id = probs.argmax().item()
    return pred_id, probs[pred_id].item(), probs.cpu().numpy()

names = ["Original", "Horizontal Flip", "Vertical Flip"]
imgs = [img_orig, img_hflip, img_vflip]

for name, im in zip(names, imgs):
    pred_id, conf, _ = predict_image(im)
    print(f"{name}: predicted={classes[pred_id]}  confidence={conf:.4f}")

plt.figure(figsize=(10,3))
for i, (name, im) in enumerate(zip(names, imgs), start=1):
    plt.subplot(1,3,i)
    plt.imshow(im)
    plt.title(name)
    plt.axis("off")
plt.show()