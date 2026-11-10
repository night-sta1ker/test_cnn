# 小型 Systolic Array 系统与验证说明

## 系统功能

本系统计算矩阵乘法：

```text
C[M x N] = A[M x K] x B[K x N]
```

- `Sequencer.v` 是顶层控制器，保存 `M/K/N` 和 A、B 的基地址。
- `SystolicArray4x4.v` 是 4x4 脉动阵列，`SystolicPE.v` 是单个处理单元。
- A 以每拍 4 个行元素的形式输入，B 以每拍 4 个列元素的形式输入；`Sequencer` 的对齐移位寄存器将两者变换为阵列需要的对角输入时序。
- N 方向按 4 列为一个 tile 处理。相同的 B tile 会随着 A 的 M 方向 4 行 block 重放，直到得到该 N tile 的全部 C 结果。
- 控制器通过 `o_act_base_valid/o_act_base_addr` 和 `o_wgt_base_valid/o_wgt_base_addr` 向上游请求数据；数据使用 `valid/ready` 握手传输。
- `o_array_en` 使能阵列计算和排空；`o_array_clear` 用于一次运算开始时清空阵列内部状态。
- `o_psum_zero` 和 `o_psum_sel` 控制部分和的清零、选择和回写路径。
- `o_psum_valid[3:0]` 标记 4 个 `i_psum_data` 输出 lane 中当前有效的数据。其仅在 `o_array_en` 有效时对外输出。

## 控制流程

`Sequencer` 的状态机为：

1. `S_IDLE`：等待 `i_size_ld`，锁存尺寸与基地址，并请求首个 A/B 数据块。
2. `S_WGT_LEAD`：先接收 B 的首拍，为阵列中的 A/B 相位对齐建立前导。
3. `S_STREAM`：在 A、B 的 `valid/ready` 握手下送入数据；对当前 N tile 的每个 M block 重放 B。
4. `S_REPLAY_WAIT`：等待下一个 A block 与 B replay 数据准备好。
5. `S_DRAIN`：停止新的有效输入，但保持阵列与对齐流水继续运行，直到最后的部分和输出。
6. `S_DONE`：本次矩阵乘法完成，随后回到 `S_IDLE`。

结果按 N tile 顺序输出；一个 tile 内遵循 4x4 阵列的对角波前顺序。不同 M block 的波前可以重叠。

## TB 验证流程

`sequencer_tb.v` 使用两个行为级 SRAM 数组模拟上游 A/B 存储器。

1. 依据测试的 `M/K/N` 将 A、B 填入 SRAM，生成对应软件 golden 矩阵 C。
2. TB 观察 `o_act_base_valid` 与 `o_wgt_base_valid`，在可配置延迟后从对应基地址开始驱动 stream。
3. TB 仅在 `stream_valid && stream_ready` 时移动 SRAM 读地址，因此检查控制器能否正确反压上游。
4. TB 为每个输出预先建立坐标 FIFO。FIFO 顺序与阵列的 N-tile、M-block、对角 phase、lane 输出顺序一致。
5. 每个时钟负边沿，TB 仅在 `o_psum_valid[lane] == 1` 时采样 `i_psum_data[lane]`，写入 FIFO 指定的 `C[row][col]`。
6. 每个案例结束后，TB 比对所有 C 坐标：检查数值错误、漏采样、额外采样和输出顺序错误。
7. 成功案例不打印矩阵内容；失败时打印案例尺寸、坐标、期望值、实际值及 `valid/sel/zero` 控制信息。

## 当前回归集合

- 256 个穷举边界案例：`M,N = 1..8`，`K = 5..8`。
- 4 个基地址响应延迟案例，验证 A/B stream 反压和握手。
- 9 个中等尺寸案例，覆盖多 M block、多 N tile 和较大 K。
- 1 个连续无复位案例，检查 `S_DONE -> S_IDLE` 后再次启动。
- 64 个大尺寸案例：`M = 100..103`、`K = 36..39`、`N = 7..10`，输入值为确定性的 `0..2`。

总计 334 个案例。执行：

```bash
make sim
```

预期最后一行：

```text
TEST SUMMARY: total_cases=334 pass=334 fail=0
```
