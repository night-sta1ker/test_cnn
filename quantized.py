import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.quantization as quant

# ================= 数据 =================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=64, shuffle=False)

# ================= 模型 =================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.quant = quant.QuantStub()      # 
        self.dequant = quant.DeQuantStub()  # 

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
        x = self.quant(x)   # 🔥 关键
        x = self.net(x)
        x = self.dequant(x)
        return x


# ================= 加载模型 =================
torch.backends.quantized.engine = 'fbgemm' 
#print(torch.backends.quantized.engine)
#print(torch.backends.quantized.supported_engines)

model = SimpleCNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()



# ================= 1. 融合 =================
model_fused = torch.quantization.fuse_modules(
    model,
    [['net.0', 'net.1'], ['net.3', 'net.4']]
)

# ================= 2. qconfig =================
model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# ================= 3. prepare =================
torch.quantization.prepare(model_fused, inplace=True)

# ================= 4. calibration =================
print("Calibrating...")
with torch.no_grad():
    for i, (data, _) in enumerate(train_loader):
        model_fused(data)
        if i > 100:   # 不用全跑
            break

# ================= 5. convert =================
torch.quantization.convert(model_fused, inplace=True)
print(model_fused)

# ================= 6. 测试 =================
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model_fused(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

print(f"Quantized Accuracy: {correct / total:.4f}")

# ================= 7. 保存 =================
torch.save(model_fused.state_dict(), "model_int8.pth")
torch.save(model_fused, "model_int8_full.pth")
print("Quantized model saved!")

# print(model_fused.net[0].scale)
# print(model_fused.net[0].zero_point)
# print(model_fused.net[3].scale)
# print(model_fused.net[3].zero_point)
# print(model_fused.net[7].scale)
# print(model_fused.net[7].zero_point)

# ================= 8. 加载查看数据结构 =================
# sd = torch.load("model_int8.pth", map_location="cpu")

# print(type(sd))          # 应该是 dict
# print(len(sd))           # 参数个数
# print(sd.keys())         # 所有key

# packed = sd['net.7._packed_params._packed_params']

# # 正确解包方式
# w, b = packed.__getstate__() if hasattr(packed, "__getstate__") else packed
  
# print(w.dtype)
# print("bias dtype:", b.dtype)

