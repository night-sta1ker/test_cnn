# import torch

# sd = torch.load("model_int8.pth", map_location="cpu")

# print(sd['net.0.bias'].dtype)
# print(sd['net.0.weight'].dtype)
#=============================================================================================================



# import numpy as np

# data = np.load("model_int8_pure.npz")

# for k in data.files:
#     print(k, data[k].shape, data[k].dtype)


import torch

MODEL_PATH = "model.pth"   # 改成你的文件名

obj = torch.load(MODEL_PATH, map_location="cpu")

# 兼容 state_dict 和整个 model
if isinstance(obj, dict):
    # 如果是 state_dict
    tensors = obj.items()
else:
    # 如果是整个 model
    tensors = obj.state_dict().items()

total_params = 0
total_bytes = 0

print("========== Parameter Details ==========")
print(f"{'Name':30s} {'Shape':20s} {'Dtype':10s} {'Numel':10s} {'Bytes':10s}")

for name, tensor in tensors:
    # 跳过非tensor对象（有些checkpoint里会混入别的信息）
    if not isinstance(tensor, torch.Tensor):
        continue

    numel = tensor.numel()
    bytes_ = numel * tensor.element_size()

    total_params += numel
    total_bytes += bytes_

    print(
        f"{name:30s} "
        f"{str(list(tensor.shape)):20s} "
        f"{str(tensor.dtype):10s} "
        f"{numel:<10d} "
        f"{bytes_:<10d}"
    )

# 向上取整换算成4-byte word
total_words = (total_bytes + 3) // 4

print("\n========== Summary ==========")
print(f"Total parameters        : {total_params}")
print(f"Real storage bytes      : {total_bytes}")
print(f"Equivalent 4B words     : {total_words}")
print(f"Storage KB              : {total_bytes / 1024:.2f} KB")
print(f"Storage MB              : {total_bytes / 1024 / 1024:.4f} MB")