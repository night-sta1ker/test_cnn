# Version5 Target

Version5 accelerates the fixed integer network in `../inference_pure.c`:

- Conv1: 1x28x28 -> 4x26x26, kernel 3x3, stride 1
- MaxPool1: 4x26x26 -> 4x13x13, kernel 2x2, stride 2
- Conv2: 4x13x13 -> 8x11x11, kernel 3x3, stride 1
- MaxPool2: 8x11x11 -> 8x5x5, kernel 2x2, stride 2
- FC: 200 -> 10, followed by ArgMax
- All packed weights remain in a 4KB WgtBuffer after model loading
- Input and all intermediate activations remain in a 4KB on-chip Activation SRAM
