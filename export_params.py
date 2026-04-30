# -*- coding: utf-8 -*-
import torch
import struct

# 加载模型参数
state = torch.load("model.pth", map_location="cpu")

# 定义参数导出函数
def export_params_to_bin(filename):
    with open(filename, "wb") as f:
        # 写入参数数量
        param_count = len(state)
        f.write(struct.pack('I', param_count))

        for name, tensor in state.items():
            # 写入参数名长度和名称
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)

            # 写入形状
            shape = tensor.shape
            f.write(struct.pack('I', len(shape)))
            for dim in shape:
                f.write(struct.pack('I', dim))

            # 写入数据
            data = tensor.numpy().astype('float32').flatten()
            f.write(struct.pack('I', len(data)))
            f.write(data.tobytes())

if __name__ == "__main__":
    export_params_to_bin("model_params.bin")
    print("Parameters exported to model_params.bin")