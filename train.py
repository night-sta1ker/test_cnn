import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. 数据
transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=64, shuffle=False)

# 2. 模型
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

model = SimpleCNN()

# 3. 训练配置
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 4. 训练
for epoch in range(10):
    correct = 0
    total = 0

    model.train()

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    train_acc = correct / total
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}")

#  关键：保存模型
torch.save(model.state_dict(), "./model.pth")
print("Model saved!")

