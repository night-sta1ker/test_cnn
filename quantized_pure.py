import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
import torch.quantization as quant

# ===== 加载 =====
sd = torch.load("model_int8.pth", map_location="cpu")

# ===== 输入量化参数 =====
Sx = sd['quant.scale'].item()
Zx = sd['quant.zero_point'].item()

print("Input:", Sx, Zx)


# ===== multiplier 计算 =====
def quantize_multiplier(M):
    if M == 0:
        return 0, 0
    shift = 0
    while M < 0.5:
        M *= 2
        shift += 1
    multiplier = int(round(M * (1 << 31)))
    return multiplier, shift


# ===== 通用卷积/FC处理 =====
def process_layer(weight, bias, Sx, Zx, Sout, Zout):

    # int8 权重
    Wq = weight.int_repr().numpy().astype(np.int8)

    # per-channel scale
    Sw = weight.q_per_channel_scales().numpy()

    # bias -> int32
    b = bias.detach().cpu().numpy()
    b_int32 = np.round(b / (Sx * Sw)).astype(np.int32)

    # multiplier + shift
    multiplier = []
    shift = []

    for c in range(len(Sw)):
        M = (Sx * Sw[c]) / Sout
        m, s = quantize_multiplier(M)
        multiplier.append(m)
        shift.append(s)

    return {
        "Wq": Wq,
        "b_int32": b_int32,
        "multiplier": np.array(multiplier, dtype=np.int32),
        "shift": np.array(shift, dtype=np.int32),
        "Zx": Zx,
        "Zout": Zout
    }


# ===== 从模型中取输出scale =====
# 注意：必须从 module 拿
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


model = torch.load("model_int8_full.pth", map_location="cpu")
# 这里简单用 sd 方式不行 → 推荐重新 load 模型结构


# ===== 临时手动填（你需要改成真实值）=====
# Conv1
Sout0 = model.net[0].scale
Zout0 = model.net[0].zero_point

# Conv2
Sout1 = model.net[3].scale
Zout1 = model.net[3].zero_point

# FC
Sout2 = model.net[7].scale
Zout2 = model.net[7].zero_point


# ===== Conv1 =====
layer0 = process_layer(
    sd['net.0.weight'],
    sd['net.0.bias'],
    Sx, Zx,
    Sout0, Zout0
)

# ===== Conv2 =====
layer1 = process_layer(
    sd['net.3.weight'],
    sd['net.3.bias'],
    Sout0, Zout0,
    Sout1, Zout1
)

# ===== FC =====
packed = sd['net.7._packed_params._packed_params']
if isinstance(packed, tuple):
    w, b = packed
else:
    w, b = packed.__getstate__()

layer2 = process_layer(
    w, b,
    Sout1, Zout1,
    Sout2, Zout2
)


# ===== 保存 =====
np.savez("model_int8_pure.npz",

    in_scale=Sx,
    in_zp=Zx,

    # conv0
    c0_w=layer0["Wq"],
    c0_b=layer0["b_int32"],
    c0_m=layer0["multiplier"],
    c0_s=layer0["shift"],
    c0_Zx=layer0["Zx"],
    c0_Zo=layer0["Zout"],

    # conv1
    c1_w=layer1["Wq"],
    c1_b=layer1["b_int32"],
    c1_m=layer1["multiplier"],
    c1_s=layer1["shift"],
    c1_Zx=layer1["Zx"],
    c1_Zo=layer1["Zout"],

    # fc
    fc_w=layer2["Wq"],
    fc_b=layer2["b_int32"],
    fc_m=layer2["multiplier"],
    fc_s=layer2["shift"],
    fc_Zx=layer2["Zx"],
    fc_Zo=layer2["Zout"],
)

print("✅ Export done: model_int8_pure.npz")

def dump_scalar(f, name, value, dtype):
    if dtype == "float":
        f.write(f"const float {name} = {float(value)}f;\n")
    else:
        f.write(f"const {dtype} {name} = {int(value)};\n")

def dump_array(f, name, arr, dtype):
    flat = arr.flatten()
    f.write(f"const {dtype} {name}[{len(flat)}] = {{\n")
    for i, v in enumerate(flat):
        if dtype == "int8_t":
            f.write(f"{int(v)},")
        else:
            f.write(f"{int(v)},")
        if (i + 1) % 16 == 0:
            f.write("\n")
    f.write("\n};\n\n")


data = np.load("model_int8_pure.npz")

with open("model_params.h", "w") as f:

    f.write("#include <stdint.h>\n\n")

    # ===== 输入量化参数 =====
    dump_scalar(f, "in_scale", data["in_scale"], "float")
    dump_scalar(f, "in_zp", data["in_zp"], "int32_t")
    f.write("\n")

    # ===== conv0 =====
    dump_array(f, "c0_w", data["c0_w"], "int8_t")
    dump_array(f, "c0_b", data["c0_b"], "int32_t")
    dump_array(f, "c0_m", data["c0_m"], "int32_t")
    dump_array(f, "c0_s", data["c0_s"], "int32_t")

    dump_scalar(f, "c0_Zx", data["c0_Zx"], "int32_t")
    dump_scalar(f, "c0_Zo", data["c0_Zo"], "int32_t")
    f.write("\n")

    # ===== conv1 =====
    dump_array(f, "c1_w", data["c1_w"], "int8_t")
    dump_array(f, "c1_b", data["c1_b"], "int32_t")
    dump_array(f, "c1_m", data["c1_m"], "int32_t")
    dump_array(f, "c1_s", data["c1_s"], "int32_t")

    dump_scalar(f, "c1_Zx", data["c1_Zx"], "int32_t")
    dump_scalar(f, "c1_Zo", data["c1_Zo"], "int32_t")
    f.write("\n")

    # ===== fc =====
    dump_array(f, "fc_w", data["fc_w"], "int8_t")
    dump_array(f, "fc_b", data["fc_b"], "int32_t")
    dump_array(f, "fc_m", data["fc_m"], "int32_t")
    dump_array(f, "fc_s", data["fc_s"], "int32_t")

    dump_scalar(f, "fc_Zx", data["fc_Zx"], "int32_t")
    dump_scalar(f, "fc_Zo", data["fc_Zo"], "int32_t")

print("✅ model_params.h generated")