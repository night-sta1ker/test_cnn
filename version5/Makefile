TOP      := sequencer_tb
SIMV     := simv
SIMV_DBG := simv_fsdb
SIMV_IM2COL := simv_im2col
SIMV_ACT_SRAM := simv_act_sram
SIMV_WGT := simv_wgt
SIMV_ACT := simv_act
SIMV_OPS := simv_ops
SIMV_CTRL := simv_ctrl
SIMV_POST := simv_post
SIMV_CORE := simv_core
SIMV_PE   := simv_pe
FSDB     := wave.fsdb

LM_LICENSE_FILE ?= 27000@fzh
MNIST_INDEX ?= 0
MNIST_CASES ?= 10
VERDI_HOME ?= /home/synopsys/verdi/Verdi_O-2018.09-SP2
NOVAS_PLI := $(VERDI_HOME)/share/PLI/VCS/linux64

VCS_FLAGS := -full64 +v2k -sverilog -timescale=1ns/1ps -licwait 5 \
	-debug_access+all -kdb \
	+lint=TFIPC-L \
	-f filelist.f \
	-top $(TOP) \
	-LDFLAGS "-Wl,--no-as-needed"

FSDB_FLAGS := +define+FSDB -P $(NOVAS_PLI)/novas.tab $(NOVAS_PLI)/pli.a

.PHONY: all regression sim fsdb buffer_check im2col_check act_sram_check wgt_buffer_check act_buffer_check pe_check operator_check controller_check postprocess_check core_check model_pack e2e_vectors verdi clean

fastcheck: clean fsdb verdi

all: sim

regression: act_sram_check buffer_check im2col_check pe_check operator_check \
	controller_check postprocess_check sim core_check

sim:
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		vcs $(VCS_FLAGS) -l compile.log -o $(SIMV)
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) ./$(SIMV) -l sim.log

fsdb:
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		vcs $(VCS_FLAGS) $(FSDB_FLAGS) -l compile_fsdb.log -o $(SIMV_DBG)
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) ./$(SIMV_DBG) +fsdbfile+$(FSDB) -l sim_fsdb.log

buffer_check: act_buffer_check wgt_buffer_check

im2col_check:
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		vcs -full64 +v2k -sverilog -timescale=1ns/1ps -licwait 5 \
		-LDFLAGS "-Wl,--no-as-needed" -Mdir=csrc_im2col \
		Im2Col3x3.v im2col_tb.v -l compile_im2col.log -o $(SIMV_IM2COL)
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) ./$(SIMV_IM2COL) -l sim_im2col.log

act_sram_check:
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		vcs -full64 +v2k -sverilog -timescale=1ns/1ps -licwait 5 \
		-LDFLAGS "-Wl,--no-as-needed" -Mdir=csrc_act_sram \
		ActivationSram.v activation_sram_tb.v -l compile_act_sram.log -o $(SIMV_ACT_SRAM)
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) ./$(SIMV_ACT_SRAM) -l sim_act_sram.log

wgt_buffer_check:
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		vcs -full64 +v2k -sverilog -timescale=1ns/1ps -licwait 5 \
		-LDFLAGS "-Wl,--no-as-needed" -Mdir=csrc_wgt \
		WgtBuffer.v wgt_buffer_tb.v -l compile_wgt.log -o $(SIMV_WGT)
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) ./$(SIMV_WGT) -l sim_wgt.log

act_buffer_check:
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		vcs -full64 +v2k -sverilog -timescale=1ns/1ps -licwait 5 \
		-LDFLAGS "-Wl,--no-as-needed" -Mdir=csrc_act \
		ActivationSram.v ActBuffer.v act_buffer_tb.v -l compile_act.log -o $(SIMV_ACT)
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) ./$(SIMV_ACT) -l sim_act.log

pe_check:
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		vcs -full64 +v2k -sverilog -timescale=1ns/1ps -licwait 5 \
		-LDFLAGS "-Wl,--no-as-needed" -Mdir=csrc_pe \
		SystolicPE.v systolic_pe_tb.v -l compile_pe.log -o $(SIMV_PE)
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) ./$(SIMV_PE) -l sim_pe.log

operator_check:
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		vcs -full64 +v2k -sverilog -timescale=1ns/1ps -licwait 5 \
		-LDFLAGS "-Wl,--no-as-needed" -Mdir=csrc_ops \
		ActivationSram.v MaxPool2x2.v ArgMax10.v pool_argmax_tb.v \
		-l compile_ops.log -o $(SIMV_OPS)
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) ./$(SIMV_OPS) -l sim_ops.log

controller_check:
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		vcs -full64 +v2k -sverilog -timescale=1ns/1ps -licwait 5 \
		-LDFLAGS "-Wl,--no-as-needed" -Mdir=csrc_ctrl \
		LayerController.v layer_controller_tb.v -l compile_ctrl.log -o $(SIMV_CTRL)
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) ./$(SIMV_CTRL) -l sim_ctrl.log

postprocess_check:
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		vcs -full64 +v2k -sverilog -timescale=1ns/1ps -licwait 5 \
		-LDFLAGS "-Wl,--no-as-needed" -Mdir=csrc_post \
		ActivationSram.v ParamBuffer.v PostProcessWriter.v postprocess_writer_tb.v \
		-l compile_post.log -o $(SIMV_POST)
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) ./$(SIMV_POST) -l sim_post.log

e2e_vectors: gen_e2e_vectors.c ../model_params.h
	$(CC) -std=c99 -O2 -Wall -Wextra -o gen_e2e_vectors gen_e2e_vectors.c -lm
	./gen_e2e_vectors $(MNIST_INDEX) $(MNIST_CASES)

core_check: model_pack e2e_vectors
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		vcs -full64 +v2k -sverilog -timescale=1ns/1ps -licwait 5 \
		-LDFLAGS "-Wl,--no-as-needed" -Mdir=csrc_core \
		Sequencer.v SystolicArray4x4.v SystolicPE.v ActivationSram.v \
		ActBuffer.v WgtBuffer.v ParamBuffer.v PostProcessWriter.v \
		MaxPool2x2.v ArgMax10.v LayerController.v NpuInferenceCore.v \
		npu_inference_core_tb.v -top npu_inference_core_tb \
		+define+E2E_FIRST_INDEX=$(MNIST_INDEX) +define+E2E_CASES=$(MNIST_CASES) \
		-l compile_core.log -o $(SIMV_CORE)
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) ./$(SIMV_CORE) -l sim_core.log

model_pack: pack_model.c ../model_params.h
	$(CC) -std=c99 -O2 -Wall -Wextra -o pack_model pack_model.c
	./pack_model

verdi: 
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		verdi -sv -f filelist.f -top $(TOP) -ssf $(FSDB) -nologo

clean:
	rm -rf csrc csrc_im2col csrc_act_sram csrc_wgt csrc_act csrc_pe csrc_ops csrc_ctrl csrc_post csrc_core \
		$(SIMV) $(SIMV).daidir $(SIMV_DBG) $(SIMV_DBG).daidir \
		$(SIMV_IM2COL) $(SIMV_IM2COL).daidir \
		$(SIMV_ACT_SRAM) $(SIMV_ACT_SRAM).daidir \
		$(SIMV_WGT) $(SIMV_WGT).daidir \
		$(SIMV_ACT) $(SIMV_ACT).daidir \
		$(SIMV_PE) $(SIMV_PE).daidir \
		$(SIMV_OPS) $(SIMV_OPS).daidir \
		$(SIMV_CTRL) $(SIMV_CTRL).daidir \
		$(SIMV_POST) $(SIMV_POST).daidir \
		$(SIMV_CORE) $(SIMV_CORE).daidir \
		pack_model gen_e2e_vectors ucli.key vc_hdrs.h novas.conf verdiLog DVEfiles \
		compile*.log *.fsdb *.vpd
		
