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

# 2. 模型（最小CNN）
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),   # 28 -> 26
            nn.ReLU(),
            nn.MaxPool2d(2),          # 26 -> 13
            nn.Conv2d(16, 32, 3, 1),  # 13 -> 11
            nn.ReLU(),
            nn.MaxPool2d(2),          # 11 -> 5
            nn.Flatten(),
            nn.Linear(32*5*5, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN()

# 3. 训练配置
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 4. 训练&测试
for epoch in range(3):
    correct = 0
    total = 0

    # ===== 训练 =====
    model.train()  # 明确进入训练模式

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    train_acc = correct / total

    # ===== 测试 =====
    correct = 0
    total = 0

    model.eval()  # 切换到测试模式

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)

            correct += (pred == target).sum().item()
            total += target.size(0)

    test_acc = correct / total

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


######################################################单独可视化####################################################################################
import matplotlib.pyplot as plt

# # 从测试集中取一批数据
data_iter = iter(test_loader)
images, labels = next(data_iter)

# # 选一张图
# img = images[0]
# label = labels[0]

# # 模型预测
# model.eval()
# with torch.no_grad():
#     output = model(img.unsqueeze(0))  # 增加batch维度
#     pred = output.argmax(dim=1).item()

# # 可视化
# plt.imshow(img.squeeze(), cmap='gray')
# plt.title(f"True: {label}, Pred: {pred}")
# plt.axis('off')
# plt.show()

fig, axes = plt.subplots(1, 6, figsize=(12, 2))

for i in range(6):
    img = images[i]
    label = labels[i]

    with torch.no_grad():
        output = model(img.unsqueeze(0))
        pred = output.argmax(dim=1).item()

    axes[i].imshow(img.squeeze(), cmap='gray')
    axes[i].set_title(f"T:{label}\nP:{pred}")
    axes[i].axis('off')

plt.show()