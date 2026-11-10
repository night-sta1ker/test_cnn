# RTL Optimization Report

## Measurement limits

No target standard-cell Liberty/DB is available in the current environment.
The numbers below are technology-independent Yosys/ABC AIG measurements, so
they are useful for relative comparison but are not frequency or slack claims.

## Baseline critical paths

| Logic cone | AIG AND nodes | AIG levels |
|---|---:|---:|
| One systolic PE: signed 8x8 multiply plus 32-bit accumulate | 1061 | 47 |
| One original requant lane: bias add, 32x16 multiply, variable shift and clamp | 7464 | 134 |

The original requant lane was the first timing target. It placed a 32-bit add,
a signed multiplier, a variable arithmetic shifter and saturation logic in one
register-to-register path.

The PostProcessWriter also contains a resource problem: its 16-entry queue can
accept four dynamic writes per cycle. It therefore does not infer a normal
single- or dual-port memory and is expanded into registers and muxes. The
baseline RTL synthesis contained 803 mux cells in this module.

## Resource hotspots

| Block | Nominal storage/compute | Main implementation concern |
|---|---:|---|
| ActivationSram | 1024 x 32 = 32 Kbit | Bind to a byte-write SRAM macro |
| WgtBuffer | 1024 x 32 = 32 Kbit | Bind to a 1R1W/dual-port SRAM macro |
| ActBuffer line memories | 4 x 32 x 32 = 4 Kbit | Four independent 32-bit banks |
| ActBuffer stream FIFO | 128 x 32 = 4 Kbit | SRAM-friendly, but read control has bubbles |
| ParamBuffer | 22 x 65 = 1430 bit | Small register file is sufficient |
| PostProcessWriter queue | 16 x 46 = 736 bit | Four dynamic writes prevent SRAM inference |
| Systolic array | 16 signed 8x8 MAC lanes | Main arithmetic area and secondary timing path |
| Requantizer | 4 signed 32x16 multipliers plus shifters | Main post-processing area and original critical path |

The largest memory arrays are structurally suitable for SRAM macros. The first
resource optimization should instead target the small PostProcessWriter queue:
its port structure, rather than its 736-bit capacity, creates the large mux and
register fabric.

## Optimization 1: pipelined requantization

The requant path is now split into registered stages:

1. `psum + bias`
2. signed `32x16` multiply
3. variable arithmetic right shift
4. add output zero point, ReLU and uint8 saturation

Valid bits, per-lane M coordinates, N-tile identity and quantization parameters
advance with the data. A new N tile updates output addresses only after the old
requant pipeline has drained.

| Optimized stage | AIG AND nodes | AIG levels |
|---|---:|---:|
| Bias add | 346 | 20 |
| 32x16 multiply | 5444 | 46 |
| Variable shift | 1017 | 25 |
| Zero point, ReLU and saturation | 637 | 55 |

The isolated requant depth dropped from 134 to a maximum of 55 AIG levels, a
59% reduction. The change adds about 1000 pipeline register bits. On the MNIST
end-to-end test, total latency increased from 24023 to 24029 cycles, or 0.025%.

## Verification

- `make postprocess_check`: PASS
- `make core_check MNIST_INDEX=0`: all Conv1, Pool1, Conv2, Pool2 and FC bytes
  match the C golden model; class 7, score 102
- No writer overflow or protocol error was observed.

## Next optimization priorities

1. Replace the four-write central queue with four single-write lane FIFOs and a
   one-write-per-cycle arbiter. This removes the multiwrite register mux fabric
   and gives a clean SRAM/register-file implementation boundary.
2. Add prefetch to the ActBuffer FIFO read side. The current synchronous reader
   waits until the output word is consumed before requesting the next word, so
   it cannot sustain one activation word per cycle.
3. Re-evaluate the 8x8 multiply-plus-accumulate PE path against a real standard
   cell or DSP library. Pipeline or macro selection should be driven by the
   target clock and library report, because changing PE latency also changes the
   Sequencer schedule.
4. Narrow fixed-network counters and metadata widths, especially M coordinates,
   N tile indices and requant shift fields, after the interface widths are
   frozen.
5. Bind ActivationSram, WgtBuffer and ActBuffer line/FIFO memories to real SRAM
   macros for ASIC implementation. Behavioral arrays alone do not prove macro
   timing, area or port legality.
