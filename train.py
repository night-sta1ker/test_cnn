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

# 2. 模型 — 方案B：极小网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1),    # 28 -> 26
            nn.ReLU(),
            nn.MaxPool2d(2),           # 26 -> 13
            nn.Conv2d(4, 8, 3, 1),    # 13 -> 11
            nn.ReLU(),
            nn.MaxPool2d(2),           # 11 -> 5
            nn.Flatten(),
            nn.Linear(8*5*5, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN()

# 参数量统计
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total}")

# 3. 训练配置
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 4. 训练 + 测试
for epoch in range(20):
    # ===== 训练 =====
    model.train()
    train_correct = 0
    train_total = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        train_correct += (pred == target).sum().item()
        train_total += target.size(0)

    train_acc = train_correct / train_total

    # ===== 测试 =====
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += (pred == target).sum().item()
            test_total += target.size(0)

    test_acc = test_correct / test_total

    print(f"Epoch {epoch:2d}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# 保存模型
torch.save(model.state_dict(), "./model.pth")
print("Model saved!")
