# -*- coding: utf-8 -*-
import torch
import struct

# Load model parameters
state = torch.load("model.pth", map_location="cpu")

# Export parameters to binary file
def export_params_to_bin(filename):
    with open(filename, "wb") as f:
        # Write parameter count
        param_count = len(state)
        f.write(struct.pack('I', param_count))

        for name, tensor in state.items():
            # Write name length and name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)

            # Write shape
            shape = tensor.shape
            f.write(struct.pack('I', len(shape)))
            for dim in shape:
                f.write(struct.pack('I', dim))

            # Write data
            data = tensor.numpy().astype('float32').flatten()
            f.write(struct.pack('I', len(data)))
            f.write(data.tobytes())

if __name__ == "__main__":
    export_params_to_bin("model_params.bin")
    print("Parameters exported to model_params.bin")
