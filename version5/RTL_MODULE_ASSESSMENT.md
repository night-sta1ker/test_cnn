# RTL Module Assessment

## Acceptance boundary

Version5 must execute the fixed network in `../inference_pure.c`, keep weights
and activations on chip, preserve all valid/ready data, and match the C golden
outputs for Conv1, Pool1, Conv2, Pool2, FC, and ArgMax. Generic layer
programmability is outside this version.

## Final verification

- `make regression MNIST_INDEX=0` passes every focused RTL test.
- Sequencer matrix regression passes 334/334 shapes, including source and
  result-path backpressure.
- PostProcessWriter passes with a forced 24-cycle write-port stall and a
  four-entry-per-lane queue.
- End-to-end inference passes Conv1, Pool1, Conv2, Pool2, FC, and ArgMax:
  class 7, score 102, 19683 cycles.
- Yosys 0.9 parses and elaborates the complete hierarchy. Its `check` pass
  reports the unpacked `psum_v` array as undriven; VCS elaboration and all 334
  numerical array cases verify those generated PE connections directly.

## Module findings

| Module | Finding | Resolution | Status |
|---|---|---|---|
| NpuInferenceCore | Busy-time preload writes could corrupt resident state; errors were transient | Block all preload writes while busy and latch protocol errors | Resolved |
| LayerController | Fixed schedule and one-cycle start/done pulses | Retained intentionally; `start` is accepted only when `busy=0` | Accepted |
| ActivationSram | Target macro collision behavior is unspecified | Fixed schedule uses disjoint source/destination regions | Accepted |
| ActBuffer | Synchronous FIFO reads inserted avoidable bubbles | Added two-entry front prefetch while preserving the DPRAM FIFO | Resolved |
| WgtBuffer | One outstanding request; writes must not overlap inference | Top-level write protection enforces this fixed schedule | Accepted |
| ParamBuffer | Four combinational reads implement a small register file | Retained for only 22 parameter entries; writes are protected at top | Accepted |
| Sequencer | No result `ready`; stale implicit net | Added result backpressure through all lockstep state and removed stale logic | Resolved |
| SystolicPE | `clear` was ignored when `en=0` | `clear` now has priority over clock enable; focused TB added | Resolved |
| SystolicArray4x4 | All lanes must freeze together | Sequencer now freezes array, aligners, valid timing, and coordinates together | Resolved |
| PostProcessWriter | Pipeline could not stall; central queue required four writes/cycle | Added `psum_ready`, stallable pipeline, four lane FIFOs, and round-robin writer | Resolved |
| MaxPool2x2 | Serial four-read implementation | Retained; fixed-network throughput and numerical TB pass | Accepted |
| ArgMax10 | Fixed ten-class scan | Retained because the target network has ten classes | Accepted |
| Im2Col3x3 | Standalone legacy module is not in production path | Retained only as a reference test; ActBuffer owns active im2col | Accepted |
| Makefile/TB | Stale buffer target and ambiguous simulation top | Buffer target uses focused tests; explicit top and `regression` target added | Resolved |

## Remaining implementation boundary

The RTL uses behavioral SRAM/DPRAM models. Foundry or FPGA memory macro
binding, timing constraints, CDC, and a system-bus wrapper are physical
integration work and are outside the Version5 fixed-inference acceptance
boundary.
