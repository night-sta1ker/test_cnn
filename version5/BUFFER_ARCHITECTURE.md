# Version5 inference_pure 加速架构

## 1. 最终目标

Version5只服务上级目录中的 `inference_pure.c`，不先实现通用CNN。

CPU负责：

- 读取MNIST文件；
- 按现有程序完成输入归一化和uint8量化；
- 将每张图的784 Bytes输入写入NPU；
- 启动NPU并读取最终分类结果。

NPU负责完整整数网络：

```text
Input  1x28x28
  -> Conv1 1x28x28 -> 4x26x26
  -> ReLU
  -> MaxPool 2x2, stride 2 -> 4x13x13
  -> Conv2 4x13x13 -> 8x11x11
  -> ReLU
  -> MaxPool 2x2, stride 2 -> 8x5x5
  -> FC 200 -> 10
  -> ArgMax -> class 0..9
```

卷积和FC复用现有Sequencer与4x4脉动阵列。权重在运行测试集前一次性
装入WgtBuffer；输入和全部中间activation始终位于NPU片上Activation SRAM。

## 2. 系统结构

```text
                        +----------------------+
CPU/DMA -- input ------>| Activation SRAM 4KB |<-------------------+
                        +----------+-----------+                    |
                                   | read                            | write
                                   v                                 |
                            +-------------+                    +-----+------+
                            | ActBuffer   |-- act stream --+    | PostProcess|
                            | conv/im2col |                |    | bias/quant |
                            | FC bypass   |                |    | ReLU/write |
                            +-------------+                v    +-----+------+
                                                    +-----------+    ^
model load -> WgtBuffer 4KB -- wgt stream --------->| Sequencer |----+
                                                    +-----+-----+
                                                          |
                                                          v
                                                   4x4 Systolic Array

Activation SRAM <-------------- MaxPool 2x2 engine
Activation SRAM --------------> ArgMax/Result register
```

新增的LayerController只执行固定五阶段流程：

```text
CONV1 -> POOL1 -> CONV2 -> POOL2 -> FC -> ARGMAX -> DONE
```

不引入指令集、描述符链或通用算子调度器。

## 3. 片上Activation SRAM

### 3.1 结构

使用一个 `1024 x 32-bit`、带byte-write-enable的1读1写DPRAM，共4KB。

- 读端口：ActBuffer、MaxPool、ArgMax分时使用；
- 写端口：PostProcess和MaxPool分时使用；
- 卷积时ActBuffer读输入，PostProcess同时写输出；
- 不向片外写回中间activation。

activation统一采用C程序相同的紧密CHW uint8布局：

```text
byte_addr(c, h, w) = base + c*H*W + h*W + w
```

### 3.2 A/B区域复用

```text
Region A: byte 0x000 .. 0x3ff，容量1024 Bytes
Region B: byte 0x400 .. 0xfff，容量3072 Bytes
```

| 阶段 | 输入区域 | 输出区域 | 有效数据量 |
| --- | --- | --- | ---: |
| 装载输入 | CPU -> A | A | 784 B |
| Conv1 | A | B | 2704 B |
| Pool1 | B | A | 676 B |
| Conv2 | A | B | 968 B |
| Pool2 | B | A | 200 B |
| FC | A | B | 10 B |

最大同时占用为输入784 B加Conv1输出2704 B，共3488 B，因此4KB足够。

## 4. ActBuffer

### 4.1 功能

ActBuffer不保存完整activation；完整activation已经在片上Activation SRAM。
它负责：

1. 接收LayerController配置和Sequencer的一拍activation请求；
2. 从Activation SRAM读取当前层输入；
3. 卷积模式下维护line buffer并生成im2col；
4. FC模式下直接生成向量流；
5. 用FIFO吸收Sequencer反压。

ActBuffer支持两个固定模式：

```text
MODE_CONV3X3
MODE_FC
```

### 4.2 uint8到signed-int8转换

C程序计算：

```text
sum((x - Zx) * w) + bias
```

片上activation是uint8，而现有阵列activation端口是signed int8。所有层统一使用：

```text
x_hw = x - 128
```

在硬件中等价于翻转uint8数据的最高位，不需要减法器。模型装载时预先修改bias：

```text
bias_hw[oc] = bias_sw[oc] + (128 - Zx) * sum(weight[oc][:])
```

于是：

```text
sum((x - 128) * w) + bias_hw
= sum((x - Zx) * w) + bias_sw
```

该变换已经用全部10000张MNIST测试图逐层比较，Conv1、Pool1、Conv2、Pool2
和FC输出均为0 mismatch。这样无需修改现有8-bit阵列数据宽度。

### 4.3 卷积line buffer

最大输入通道数为4，使用4个通道bank。每个bank是 `32 x 32-bit` 1R1W
DPRAM，内部保存4行、每行最多8个word：

```text
line_mem[channel][row_slot][word]
channel  = 0..3
row_slot = 0..3
word     = 0..7
```

总line-buffer容量512 Bytes。

计算输出行y时：

- 读取y、y+1、y+2三行；
- 同时将y+3行写入第4个空闲row slot；
- 当前输出行生成完成后逻辑旋转row slot；
- `row_valid + row_tag`保证未填满的三行绝不启动输出。

每个通道bank的读写端口可并行工作。Conv1只启用bank0；Conv2启用全部4个bank。

### 4.4 im2col输出

对于每个 `(channel, kh, x_group)`：

1. 连续两拍读取相邻两个32-bit line word；
2. 缓存成64-bit；
3. 从缓存产生kw=0、1、2三个32-bit输出；
4. 后两个kw周期不再访问line DPRAM。

输出顺序固定为：

```text
out_row -> x_group -> channel -> kh -> kw
```

一个word的4个lane对应同一行的4个相邻输出像素。行尾lane补零，不跨到下一行。

卷积映射到Sequencer：

```text
OH       = H - 2
OW       = W - 2
X_GROUPS = ceil(OW / 4)
M_SEQ    = OH * X_GROUPS * 4
K_SEQ    = 9 * C
N_SEQ    = output channels
```

### 4.5 FC模式

FC输入是Pool2产生的200个CHW uint8值。配置：

```text
M_SEQ = 1
K_SEQ = 200
N_SEQ = 10
```

每个activation stream word为：

```text
{8'sd0, 8'sd0, 8'sd0, input[k] - 128}
```

只使用M方向lane0，另外3个lane补零。FC的三个N tile会使Sequencer请求三次
activation；ActBuffer每次从片上SRAM地址0重新流出同一200 Bytes向量。

### 4.6 FIFO与反压

保留一个 `128 x 32-bit` 1R1W DPRAM FIFO，共512 Bytes。

- FIFO满：停止im2col/FC生成；
- FIFO空：`act_stream_valid=0`；
- `valid && !ready`：输出data和valid保持不变；
- FIFO空间释放后继续原计数位置；
- clear同时清空FIFO计数、输出寄存器、pending read、line tag和所有状态机。

Sequencer的base request没有ready，因此ActBuffer必须用一项pending寄存器捕获请求。
Conv2有两个N tile，第二次请求只会从片上Activation SRAM重做im2col，不访问片外。

## 5. WgtBuffer

### 5.1 全模型驻留

WgtBuffer使用 `1024 x 32-bit` SRAM，共4KB。模型启动时一次性装载，此后处理
10000张图时不再读取片外权重。

原始权重：

```text
Conv1: 4 * 1 * 3 * 3 =   36 Bytes
Conv2: 8 * 4 * 3 * 3 =  288 Bytes
FC:   10 * 200       = 2000 Bytes
```

为匹配阵列，每个32-bit word打包4个输出通道。FC最后一个N tile有两个无效
输出通道，需要补零，因此实际打包后为：

```text
Conv1: 1 tile *   9 K words =   9 words
Conv2: 2 tiles *  36 K words =  72 words
FC:    3 tiles * 200 K words = 600 words
----------------------------------------
Total                           681 words = 2724 Bytes
```

### 5.2 固定地址表

```text
word   0 ..   8 : Conv1, base=0,  K=9,   N=4
word   9 ..  44 : Conv2 tile0
word  45 ..  80 : Conv2 tile1, base自动增加36
word  81 .. 280 : FC tile0
word 281 .. 480 : FC tile1, base自动增加200
word 481 .. 680 : FC tile2
```

LayerController启动层时只需设置：

```text
Conv1: wgt_base=0,  K=9,   N=4
Conv2: wgt_base=9,  K=36,  N=8
FC:    wgt_base=81, K=200, N=10
```

现有Sequencer会在新N tile时自动把base增加K，地址表与当前控制方式一致。

### 5.3 权重重放

WgtBuffer不再包含global-buffer refill状态机。`base_valid`到来时：

1. 捕获base和K；
2. 从本地Wgt SRAM开始读；
3. 输出K个word；
4. Sequencer为下一个M block再次请求相同base时，读指针直接归位重放。

输出寄存器在`valid && !ready`时保持，WgtBuffer不需要额外FIFO。

### 5.4 量化参数

WgtBuffer内部另设小型ParamBuffer，保存22个输出通道的：

```text
corrected_bias[31:0]
multiplier[15:0]
shift[7:0]
Zout[7:0]
relu_enable
```

参数容量小于256 Bytes。参数在模型装载阶段与权重一起写入，层启动时送到
PostProcess。corrected_bias按4.2节公式离线生成。

## 6. PostProcess与结果写回

现有阵列只产生32-bit累加值，而C程序还执行bias、requant和ReLU，因此需要
PostProcess：

```text
acc1 = psum + corrected_bias[oc]
tmp  = acc1 * multiplier[oc]
q    = tmp >>> (15 + shift[oc])
q    = q + Zout
q    = clamp(q, 0, 255)
Conv层: q = max(q, Zout)
FC层:   不执行ReLU
```

Sequencer结果端没有ready，PostProcess必须每拍接住所有有效lane。Version5采用
4个并行后处理lane，每路按实际byte地址将连续结果拼成32-bit写word。

四路可能在同一拍同时拼满，因此在SRAM写口前放置一个16项写队列。该队列每拍
最多接收4项、向Activation SRAM提交1项；它只吸收多路word完成的瞬时峰值，
不是activation数据缓存。每路平均每4个有效结果才产生一个写word，因此持续写
带宽不超过单个32-bit写口。队列溢出会置错误标志，禁止静默丢失结果。

ResultWriter按照`N tile、M block、lane`坐标将uint8结果以byte enable写入
Activation SRAM，并丢弃每个输出行中因4-lane对齐产生的尾部padding。Conv2
的121 Bytes通道跨度和FC的1 Byte通道跨度允许跨通道共享同一个32-bit SRAM
word，由byte enable保证分拍写入时互不覆盖。

Sequencer计算逻辑保持不变，但需要导出两个控制信息：

- 当前结果对应的M位置/N tile坐标，供ResultWriter寻址；
- `done`脉冲，供LayerController切换阶段。

## 7. MaxPool与ArgMax

MaxPool固定为2x2、stride=2、floor模式，与C程序一致。为保持控制简单，先将
完整卷积输出写入Activation SRAM，再启动MaxPool。

MaxPool FSM对每个输出读取4个uint8值、做3次比较并写回另一区域。Conv2输出
是11x11，最后一行和最后一列按floor规则忽略。

FC完成后，ArgMax顺序读取10个uint8输出，保存最大值和索引，最终给CPU返回
4-bit类别和done状态。

## 8. 固定层配置

| 层 | HxW | C->OC | Sequencer M | K | N | Act输入 | Act输出 | Wgt base |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| Conv1 | 28x28 | 1->4 | 728 | 9 | 4 | A | B | 0 |
| Conv2 | 13x13 | 4->8 | 132 | 36 | 8 | A | B | 9 |
| FC | 200 | 1->10 | 1 | 200 | 10 | A | B | 81 |

Conv1真实空间点为26x26=676，M中每行补到28，共728；Conv2真实空间点为
11x11=121，M中每行补到12，共132。ResultWriter不写这些padding点。

理论MAC数量：

```text
Conv1: 4 * 26 * 26 * 9  = 24336
Conv2: 8 * 11 * 11 * 36 = 34848
FC:    10 * 200          =  2000
Total                         61184 MAC
```

## 9. 控制时序

每张图执行：

```text
1. CPU写784 Bytes到Activation SRAM Region A
2. CONV1: ActBuffer(A) + WgtBuffer(base 0) -> PostProcess -> B
3. POOL1: B -> A
4. CONV2 tile0/tile1: ActBuffer(A)重放两次 + WgtBuffer(base 9/45) -> B
5. POOL2: B -> A
6. FC tile0/tile1/tile2: ActBuffer(A)重放三次 + WgtBuffer(base 81/281/481) -> B
7. ARGMAX读取B中的10 Bytes
8. NPU置done，CPU读取类别
```

只有输入784 Bytes和最终类别跨越CPU/NPU边界；权重只装载一次，中间结果不离开
片上Activation SRAM。

## 10. 实现与验证顺序

1. 先实现4KB Activation SRAM、地址映射和CPU写入口。
2. 重写WgtBuffer为全模型驻留/本地重放，并制作权重打包文件。
3. 重写ActBuffer的CONV模式，验证Conv1 im2col数据流。
4. 接PostProcess和ResultWriter，端到端比对Conv1的2704 Bytes。
5. 加MaxPool，验证Pool1的676 Bytes。
6. 扩展ActBuffer到4通道并完成Conv2两个N tile。
7. 加FC模式、Pool2和ArgMax。
8. 对同一批输入逐层比对C程序，最后比对10000张分类结果和准确率。

在单层逐字节结果与C程序一致之前，不做跨层融合、双图并行或更大的通用配置。
