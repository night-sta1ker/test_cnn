import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 数据
transform = transforms.ToTensor()
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transform, download=True),
    batch_size=64, shuffle=False)

# 2. 模型（必须和train.py一致）
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*5*5, 10)
        )

    def forward(self, x):
        return self.net(x)

# 3. 加载模型
model = SimpleCNN()
model.load_state_dict(torch.load("./model.pth"))
model.eval()

print("Model loaded!")

# ================= 测试 =================
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)

        correct += (pred == target).sum().item()
        total += target.size(0)

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")

# ================= 可视化 =================
data_iter = iter(test_loader)
images, labels = next(data_iter)


import random

# 随机取6个样本（跨batch）
samples = random.sample(list(test_loader.dataset), 6)

fig, axes = plt.subplots(1, 6, figsize=(12, 2))

for i, (img, label) in enumerate(samples):

    with torch.no_grad():
        output = model(img.unsqueeze(0))
        pred = output.argmax(dim=1).item()

    axes[i].imshow(img.squeeze(), cmap='gray')
    axes[i].set_title(f"True:{label}\nPred:{pred}")
    axes[i].axis('off')

plt.show()