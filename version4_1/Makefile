TOP      := sequencer_tb
SIMV     := simv
SIMV_DBG := simv_fsdb
FSDB     := wave.fsdb

LM_LICENSE_FILE ?= 27000@fzh
VERDI_HOME ?= /home/synopsys/verdi/Verdi_O-2018.09-SP2
NOVAS_PLI := $(VERDI_HOME)/share/PLI/VCS/linux64

VCS_FLAGS := -full64 +v2k -sverilog -timescale=1ns/1ps -licwait 5 \
	-debug_access+all -kdb \
	+lint=TFIPC-L \
	-f filelist.f \
	-LDFLAGS "-Wl,--no-as-needed"

FSDB_FLAGS := +define+FSDB -P $(NOVAS_PLI)/novas.tab $(NOVAS_PLI)/pli.a

.PHONY: all sim fsdb verdi clean

fastcheck: clean fsdb verdi

all: sim

sim:
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		vcs $(VCS_FLAGS) -l compile.log -o $(SIMV)
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) ./$(SIMV) -l sim.log

fsdb:
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		vcs $(VCS_FLAGS) $(FSDB_FLAGS) -l compile_fsdb.log -o $(SIMV_DBG)
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) ./$(SIMV_DBG) +fsdbfile+$(FSDB) -l sim_fsdb.log

verdi: 
	env LM_LICENSE_FILE=$(LM_LICENSE_FILE) \
		verdi -sv -f filelist.f -top $(TOP) -ssf $(FSDB) -nologo

clean:
	rm -rf csrc $(SIMV) $(SIMV).daidir $(SIMV_DBG) $(SIMV_DBG).daidir \
		ucli.key vc_hdrs.h novas.conf verdiLog DVEfiles \
		*.log *.fsdb *.vpd
		
