# version4_1 VCS/Verdi Flow

This folder is a Linux/VCS working copy of `version4`.

## Files

- `filelist.f`: RTL/testbench compile list.
- `Makefile`: VCS simulation and Verdi waveform flow.
- `sequencer_tb.v`: supports optional FSDB dumping with `+define+FSDB`.

## Commands

Compile and run normal simulation:

```sh
make run
```

Compile and run with FSDB waveform dump:

```sh
make run_fsdb
```

Open Verdi after generating `wave.fsdb`:

```sh
make verdi
```

Remove generated simulation files:

```sh
make clean
```

The default license setting is `LM_LICENSE_FILE=27000@fzh`. Override it from
the command line if needed:

```sh
make run LM_LICENSE_FILE=<your-license-server>
```
