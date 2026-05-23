# version4 开发上下文记录

## 当前目标

继续维护 `version4` 中的 `Sequencer`，用于给 4x4 systolic array 产生对齐后的 `act_in`、`wgt_in`、`wgt_ld` 和 `en`。

当前重点是验证 activation 和 weight 的 stream 输入、内部延迟、排空阶段，以及 N group / M group 切换时的控制是否正确。

## 关键文件

- `version4/Sequencer.v`
- `version4/sequencer_tb.v`
- `version4/example.txt`
- `version4/transcript`

## ModelSim 命令

在 `C:\Users\ly\Desktop\test_cnn` 下运行：

```powershell
D:\modelsim\win64\vlog.exe .\version4\Sequencer.v .\version4\sequencer_tb.v
D:\modelsim\win64\vsim.exe -c sequencer_tb -l .\version4\transcript -do "run -all; quit"
```

## 当前 tb 测试 case

`sequencer_tb.v` 现在会打印每个 case 的 A/B 矩阵，方便对照数据流。

- case1: `A[5x6] * B[6x8]`
- case2: `A[2x3] * B[3x2]`
- case3: `A[8x9] * B[9x6]`

矩阵元素按自然序号填充。例如 case3:

```text
A:
1  2  3  ...  9
10 11 12 ... 18
...
64 65 66 ... 72

B:
1  2  3  4  5  6
7  8  9  10 11 12
...
49 50 51 52 53 54
```

## Stream 输入格式

Activation stream 按 M 方向 4 行一组、K 方向逐列输入。

以 `A[5x6]` 为例：

```text
1  7  13 19
2  8  14 20
3  9  15 21
4  10 16 22
5  11 17 23
6  12 18 24
25 0  0  0
26 0  0  0
27 0  0  0
28 0  0  0
29 0  0  0
30 0  0  0
```

Weight stream 按 N 方向 4 列一组、K 方向逐行输入。

以 `B[6x8]` 的前 4 列为例：

```text
1  2  3  4
9  10 11 12
17 18 19 20
25 26 27 28
33 34 35 36
41 42 43 44
```

## 当前控制约定

- `wgt` 要比 `act` 早一个 cycle 开始。
- `wgt` 在每个 M block 内需要 replay。
- 一个 N group 完成后，`act` 从 `in_act_base_addr` 重新读。
- 下一个 N group 的 `wgt_base_addr` 按 `k_size` 偏移。
- 排空阶段 `ready` 应该拉低，不应该依赖 tb 把 `valid` 拉低。
- `en` 当前用于表示 `act_in/wgt_in` 是否有效，需要继续检查和数据排空是否完全一致。

## 当前仿真状态

最近一次运行结果：

```text
vlog: Errors 0, Warnings 0
vsim: Errors 0, Warnings 0
case1 DONE at cycle 44
case2 DONE at cycle 13
case3 DONE at cycle 56
```

当前已经观察到：

```text
case3 cycle27 act_in=0,72,0,0 en=1
case3 cycle54 act_in=0,72,0,0 en=1
```

说明 case3 的 `72` 已经能在排空阶段输出，但它出现在 cycle27 / cycle54，不是 cycle29。cycle29 已经是下一个 N group 的 base reload 阶段。

## 已发现和修过的问题

- `wgt_stream_ready` 在最后一个 weight word 后曾经多保持一个 cycle，目前已让最后一个 wgt group 完成后提前拉低。
- `group_read_done` 后的排空阶段，两个 stream ready 应拉低。
- tb 之前固定 `act_words_left=12`、`wgt_words_left=6`，现在改为按矩阵尺寸计算：

```verilog
act_words_left <= k_size * ((m_size + 16'd3) >> 2);
wgt_words_left <= k_size;
```

- tb 已加入 SRAM 模型，sequencer 通过 `act_base_valid/wgt_base_valid` 和 base address 读 stream。
- `example.txt` 是 case1 的 golden reference，用来检查每个 `en=1` 的 `act_in/wgt_in`。

## 需要继续关注

1. 继续确认 `act_in` 排空阶段的 lane 顺序是否对所有 M/K 尺寸都正确。
2. 继续确认 `en` 是否应该只由 `act_out_en & wgt_out_en` 产生，还是排空阶段需要更细的定义。
3. 对照 `example.txt` 检查 case1 每个 `en=1` 的 `act_in/wgt_in`。
4. 对 case2、case3 建议也补充 golden 表，避免只靠肉眼看 transcript。
5. 后续如果要重构，优先保持控制信号数量少，避免重新引入复杂的 tag/valid 方案。

## 当前讨论结论

不要引入 `act_tag` / `act_valid` 这一类额外复杂标记。之前尝试过，但用户要求删除。当前方向应继续基于现有 lane、`lane_sequen`、`act_remain_cnt`、`wgt_remain_cnt` 和状态机信号修正数据流。

