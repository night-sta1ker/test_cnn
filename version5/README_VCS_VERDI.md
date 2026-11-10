# Version5 VCS/Verdi Flow

Version5 extends the version4_1 Sequencer and 4x4 systolic array into the
fixed CNN accelerator for `../inference_pure.c`. Conv1, Conv2 and FC share
the array; pooling and ArgMax use dedicated engines. Weights and intermediate
activations stay in on-chip memories.

## Files

- `NpuInferenceCore.v`: inference top-level module.
- `LayerController.v`: fixed Conv1/Pool1/Conv2/Pool2/FC/ArgMax schedule.
- `ActBuffer.v`: line memories, im2col generation and activation FIFO.
- `npu_inference_core_tb.v`: multi-image, per-layer golden checks.
- `filelist.f`: RTL and Sequencer testbench compile list.
- `Makefile`: VCS simulation and Verdi waveform flow.
- `sequencer_tb.v`: supports optional FSDB dumping with `+define+FSDB`.
- `pack_model.c`: required weight packing and bias correction utility.
- `gen_e2e_vectors.c`: required MNIST input/golden vector generator.
- `*.hex`: packed model and ten-image test vectors.
- `sim*.log`: retained simulation results from August 17, 2026.

Keep this directory inside the parent repository: vector generation uses
`../model_params.h` and `../data/MNIST/raw/`. See `VERSION5_TARGET.md` for
the network dimensions. Architecture and optimization documents include
historical proposals; use current RTL and logs for the implemented behavior.

## Commands

Run the full regression:

```sh
make regression
```

Run ten consecutive MNIST images, with weights loaded once:

```sh
make core_check MNIST_INDEX=0 MNIST_CASES=10
```

The retained core log reports 10/10 passing images, with 18784 inference
cycles each. Every case checks Conv1, Pool1, Conv2, Pool2 and FC bytes,
the classification score and the dataset label. This is not a full
10000-image RTL accuracy evaluation. The vector generator currently rejects
samples that the software model misclassifies.

Run only the 334-case Sequencer matrix regression:

```sh
make sim
```

Compile and run with FSDB waveform dump:

```sh
make fsdb
```

Open Verdi after generating `wave.fsdb`:

```sh
make verdi
```

Remove compiled binaries, compilation logs and waveforms, retaining
`sim*.log`, RTL, testbenches and vectors:

```sh
make clean
```

The default license setting is `LM_LICENSE_FILE=27000@fzh`. Override it from
the command line if needed:

```sh
make sim LM_LICENSE_FILE=<your-license-server>
```
