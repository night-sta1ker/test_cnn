`timescale 1ns / 1ps

module SequencerAlignCase #(
    parameter DATA_W = 16,
    parameter ACC_W  = 32,
    parameter M_SIZE = 4,
    parameter K_SIZE = 4,
    parameter N_SIZE = 4,
    parameter CASE_ID = 0,
    parameter MAX_CYCLES = 300,
    parameter WGT_STALL_START = -1,
    parameter WGT_STALL_LEN = 0
) (
    input  wire clk,
    input  wire rst_n,
    output reg  case_done,
    output reg  case_pass
);

    reg valid;
    wire ready;

    reg [4*DATA_W-1:0] act_stream_in;
    reg                act_stream_valid;
    wire               act_stream_ready;

    reg [4*DATA_W-1:0] wgt_stream_in;
    reg                wgt_stream_valid;
    wire               wgt_stream_ready;
    wire               wgt_stream_start;
    wire [15:0]        wgt_stream_n_base;
    wire               act_stream_start;
    wire [31:0]        act_base_addr;
    wire [31:0]        wgt_base_addr;
    wire               act_base_valid;
    wire               wgt_base_valid;

    wire [3:0]              wgt_ld;
    wire [4*DATA_W-1:0]     wgt_in;
    wire                    en;
    wire                    clear;
    wire [4*DATA_W-1:0]     act_in;
    wire [15:0]             psum_zero;
    wire [1:0]              psum_out_sel;
    reg  [4*ACC_W-1:0]      psum_out;
    wire                    done;
    wire [31:0]             cycle;
    wire [M_SIZE*N_SIZE*ACC_W-1:0] c_out_flat;

    integer act_next;
    integer wgt_row;
    integer wgt_block;
    integer sim_cycle;
    integer lane;

    reg signed [DATA_W-1:0] lane_val;
    reg act_ready_armed;
    reg wgt_ready_armed;
    reg align_started;
    reg wgt_one_cycle_ahead;
    reg wgt_stall_printed;

    Sequencer #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MAX_M_SIZE(M_SIZE),
        .MAX_K_SIZE(K_SIZE),
        .MAX_N_SIZE(N_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid(valid),
        .ready(ready),
        .size_ld(1'b0),
        .m_size(M_SIZE[15:0]),
        .k_size(K_SIZE[15:0]),
        .n_size(N_SIZE[15:0]),
        .act_stream_in(act_stream_in),
        .act_stream_valid(act_stream_valid),
        .act_stream_ready(act_stream_ready),
        .wgt_stream_in(wgt_stream_in),
        .wgt_stream_valid(wgt_stream_valid),
        .wgt_stream_ready(wgt_stream_ready),
        .wgt_stream_start(wgt_stream_start),
        .wgt_stream_n_base(wgt_stream_n_base),
        .act_stream_start(act_stream_start),
        .act_base_addr(act_base_addr),
        .wgt_base_addr(wgt_base_addr),
        .act_base_valid(act_base_valid),
        .wgt_base_valid(wgt_base_valid),
        .wgt_ld(wgt_ld),
        .wgt_in(wgt_in),
        .en(en),
        .clear(clear),
        .act_in(act_in),
        .psum_zero(psum_zero),
        .psum_out_sel(psum_out_sel),
        .psum_out(psum_out),
        .done(done),
        .cycle(cycle),
        .c_out_flat(c_out_flat)
    );

    task drive_act;
        integer l;
        integer value;
        begin
            act_stream_in = {4*DATA_W{1'b0}};
            for (l = 0; l < 4; l = l + 1) begin
                value = act_next + l + 1;
                if (value <= M_SIZE*K_SIZE)
                    act_stream_in[l*DATA_W +: DATA_W] = value[DATA_W-1:0];
            end
        end
    endtask

    task drive_wgt;
        integer l;
        integer col_i;
        integer value;
        begin
            wgt_stream_in = {4*DATA_W{1'b0}};
            for (l = 0; l < 4; l = l + 1) begin
                col_i = wgt_block*4 + l;
                value = 1000 + wgt_row*N_SIZE + col_i + 1;
                if ((wgt_row < K_SIZE) && (col_i < N_SIZE))
                    wgt_stream_in[l*DATA_W +: DATA_W] = value[DATA_W-1:0];
            end
        end
    endtask

    task print_vec;
        input [4*DATA_W-1:0] vec;
        begin
            for (lane = 0; lane < 4; lane = lane + 1) begin
                lane_val = $signed(vec[lane*DATA_W +: DATA_W]);
                $write(" %0d", lane_val);
            end
        end
    endtask

    task print_report_line;
        begin
            $write("CASE%0d CYCLE %0d state=%0d en=%0b clear=%0b wgt_ld=%b",
                   CASE_ID, sim_cycle, dut.state, en, clear, wgt_ld);
            $write(" | hold act=%0b/%0b wgt=%0b/%0b",
                   dut.act_trans_hold_i, dut.act_trans_hold,
                   dut.wgt_trans_hold_i, dut.wgt_trans_hold);
            $write(" | WGT");
            print_vec(wgt_in);
            $write(" | ACT");
            print_vec(act_in);
            $write(" | stream_valid act=%0b wgt=%0b", act_stream_valid, wgt_stream_valid);
            $write(" | stream_ready act=%0b wgt=%0b", act_stream_ready, wgt_stream_ready);
            $write(" | array_valid act=%0b wgt=%0b", dut.act_array_valid, dut.wgt_array_valid);
            $write(" | trans_valid act=%0b wgt=%0b", dut.act_trans_out_valid, dut.wgt_trans_out_valid);
            $write(" | base_valid act=%0b wgt=%0b", act_base_valid, wgt_base_valid);
            $write(" | base_addr act=%0d wgt=%0d", act_base_addr, wgt_base_addr);
            $write(" | start act=%0b wgt=%0b", act_stream_start, wgt_stream_start);
            $write(" | wgt_age=%0d replay_idx=%0d lead=%0d",
                   dut.wgt_age, dut.wgt_replay_idx, dut.wgt_lead_count);
            if (dut.wgt_trans_block_done)
                $write(" | WGT_BLOCK_DONE");
            $display("");
        end
    endtask

    task print_test_matrices;
        integer r;
        integer c;
        integer value;
        begin
            $display("");
            $display("CASE%0d TEST MATRICES: A[%0d x %0d], B[%0d x %0d]",
                     CASE_ID, M_SIZE, K_SIZE, K_SIZE, N_SIZE);

            $display("CASE%0d A matrix:", CASE_ID);
            for (r = 0; r < M_SIZE; r = r + 1) begin
                $write("CASE%0d   A[%0d]:", CASE_ID, r);
                for (c = 0; c < K_SIZE; c = c + 1) begin
                    value = r*K_SIZE + c + 1;
                    $write(" %0d", value);
                end
                $display("");
            end

            $display("CASE%0d B matrix:", CASE_ID);
            for (r = 0; r < K_SIZE; r = r + 1) begin
                $write("CASE%0d   B[%0d]:", CASE_ID, r);
                for (c = 0; c < N_SIZE; c = c + 1) begin
                    value = 1000 + r*N_SIZE + c + 1;
                    $write(" %0d", value);
                end
                $display("");
            end
            $display("");
        end
    endtask

    task run_case;
        begin
            valid = 1'b0;
            act_stream_in = 0;
            act_stream_valid = 1'b0;
            wgt_stream_in = 0;
            wgt_stream_valid = 1'b0;
            psum_out = 0;
            act_next = 0;
            wgt_row = 0;
            wgt_block = 0;
            sim_cycle = 0;
            case_done = 1'b0;
            case_pass = 1'b1;
            act_ready_armed = 1'b0;
            wgt_ready_armed = 1'b0;
            align_started = 1'b0;
            wgt_one_cycle_ahead = 1'b0;
            wgt_stall_printed = 1'b0;

            print_test_matrices();
            @(negedge clk);
            valid = 1'b1;
            act_stream_valid = 1'b1;
            wgt_stream_valid = 1'b1;
            drive_act();
            drive_wgt();

            @(negedge clk);
            valid = 1'b0;

            while (!done && (sim_cycle < MAX_CYCLES)) begin
                @(negedge clk);

                if (act_stream_valid && act_stream_ready && act_ready_armed) begin
                    act_next = act_next + 4;
                    if (act_next >= M_SIZE*K_SIZE) begin
                        act_stream_valid = 1'b0;
                        act_stream_in = 0;
                        act_ready_armed = 1'b0;
                    end else begin
                        drive_act();
                    end
                end
                if (act_stream_ready)
                    act_ready_armed = 1'b1;
                else
                    act_ready_armed = 1'b0;

                if (wgt_stream_valid) begin
                    wgt_row = dut.trans_wgt_u.load_idx;
                    if (wgt_row >= K_SIZE) begin
                        wgt_stream_valid = 1'b0;
                        wgt_stream_in = 0;
                    end else begin
                        drive_wgt();
                    end
                end

                if ((WGT_STALL_START >= 0) &&
                    (sim_cycle >= WGT_STALL_START) &&
                    (sim_cycle < (WGT_STALL_START + WGT_STALL_LEN))) begin
                    if (!wgt_stall_printed) begin
                        $display("CASE%0d WGT_INPUT_STALL begin at cycle %0d for %0d cycles",
                                 CASE_ID, sim_cycle, WGT_STALL_LEN);
                        wgt_stall_printed = 1'b1;
                    end
                    wgt_stream_valid = 1'b0;
                end else if ((dut.trans_wgt_u.load_idx < K_SIZE) && !wgt_stream_valid &&
                             !dut.wgt_trans_done) begin
                    wgt_stream_valid = 1'b1;
                    wgt_row = dut.trans_wgt_u.load_idx;
                    drive_wgt();
                end

                if (wgt_stream_start) begin
                    wgt_block = wgt_stream_n_base / 4;
                    wgt_row = 0;
                    wgt_ready_armed = 1'b0;
                    wgt_stream_valid = 1'b1;
                    drive_wgt();
                    $display("CASE%0d WGT_STREAM_START n_base=%0d at cycle %0d",
                             CASE_ID, wgt_stream_n_base, sim_cycle);
                end

                if (wgt_base_valid) begin
                    wgt_block = wgt_base_addr / 4;
                    wgt_row = 0;
                    wgt_ready_armed = 1'b0;
                    wgt_stream_valid = 1'b1;
                    drive_wgt();
                    $display("CASE%0d WGT_BASE_REQ base=%0d at cycle %0d",
                             CASE_ID, wgt_base_addr, sim_cycle);
                end

                if (act_stream_start && (wgt_stream_n_base != 0)) begin
                    act_next = 0;
                    act_ready_armed = 1'b0;
                    align_started = 1'b0;
                    wgt_one_cycle_ahead = 1'b0;
                    wgt_stall_printed = 1'b0;
                    act_stream_valid = 1'b1;
                    drive_act();
                    $display("CASE%0d REPLAY_ACT_FOR_NEXT_WGT_BLOCK at cycle %0d", CASE_ID, sim_cycle);
                end

                if (dut.act_array_valid && dut.wgt_array_valid)
                    align_started = 1'b1;

                if (align_started && dut.wgt_array_valid &&
                    !dut.act_array_valid) begin
                    if (wgt_one_cycle_ahead) begin
                        $display("CASE%0d ALIGN_ERROR at cycle %0d: WGT advanced more than one cycle ahead",
                                 CASE_ID, sim_cycle);
                        case_pass = 1'b0;
                    end
                    wgt_one_cycle_ahead = 1'b1;
                end else if (align_started && dut.act_array_valid &&
                    !dut.wgt_array_valid && !wgt_one_cycle_ahead &&
                    !dut.wgt_trans_block_done && !dut.wgt_block_pending &&
                    !dut.wgt_trans_done) begin
                    $display("CASE%0d ALIGN_ERROR at cycle %0d: ACT advanced without WGT",
                             CASE_ID, sim_cycle);
                    case_pass = 1'b0;
                end else if (dut.act_array_valid) begin
                    wgt_one_cycle_ahead = 1'b0;
                end

                print_report_line();

                sim_cycle = sim_cycle + 1;
            end

            if (!done) begin
                $display("CASE%0d TIMEOUT after %0d cycles", CASE_ID, sim_cycle);
                $display("  state=%0d act_done=%0b wgt_done=%0b wgt_block=%0d act_next=%0d wgt_row=%0d",
                         dut.state, dut.act_trans_done, dut.wgt_trans_done,
                         wgt_block, act_next, wgt_row);
                $display("  act_state=%0d act_load_m=%0d act_load_k=%0d act_launch_m=%0d act_out_cycle=%0d",
                         dut.trans_act_u.state, dut.trans_act_u.load_m,
                         dut.trans_act_u.load_k, dut.trans_act_u.launch_m,
                         dut.trans_act_u.out_cycle);
                $display("  act_row_stream=%0b%0b%0b%0b%0b%0b%0b%0b rd_valid=%0b%0b%0b%0b%0b%0b%0b%0b",
                         dut.trans_act_u.row_stream[7], dut.trans_act_u.row_stream[6],
                         dut.trans_act_u.row_stream[5], dut.trans_act_u.row_stream[4],
                         dut.trans_act_u.row_stream[3], dut.trans_act_u.row_stream[2],
                         dut.trans_act_u.row_stream[1], dut.trans_act_u.row_stream[0],
                         dut.trans_act_u.rd_valid_q[7], dut.trans_act_u.rd_valid_q[6],
                         dut.trans_act_u.rd_valid_q[5], dut.trans_act_u.rd_valid_q[4],
                         dut.trans_act_u.rd_valid_q[3], dut.trans_act_u.rd_valid_q[2],
                         dut.trans_act_u.rd_valid_q[1], dut.trans_act_u.rd_valid_q[0]);
                case_pass = 1'b0;
            end else begin
                $display("CASE%0d DONE at cycle %0d", CASE_ID, sim_cycle);
            end

            case_done = 1'b1;
        end
    endtask

    initial begin
        valid = 1'b0;
        act_stream_in = 0;
        act_stream_valid = 1'b0;
        wgt_stream_in = 0;
        wgt_stream_valid = 1'b0;
        psum_out = 0;
        act_next = 0;
        wgt_row = 0;
        wgt_block = 0;
        sim_cycle = 0;
        case_done = 1'b0;
        case_pass = 1'b1;
        act_ready_armed = 1'b0;
        wgt_ready_armed = 1'b0;
        align_started = 1'b0;
        wgt_one_cycle_ahead = 1'b0;
        wgt_stall_printed = 1'b0;

        wait (rst_n == 1'b1);
    end

endmodule

module sequencer_align_tb;

    reg clk;
    reg rst_n;

    wire done0;
    wire done1;
    wire done2;
    wire pass0;
    wire pass1;
    wire pass2;

    integer tb_cycle;
    reg run0;
    reg run1;
    reg run2;

    SequencerAlignCase #(
        .M_SIZE(2),
        .K_SIZE(3),
        .N_SIZE(2),
        .CASE_ID(0),
        .MAX_CYCLES(120)
    ) case_2x3_3x2 (
        .clk(clk),
        .rst_n(rst_n),
        .case_done(done0),
        .case_pass(pass0)
    );

    SequencerAlignCase #(
        .M_SIZE(4),
        .K_SIZE(4),
        .N_SIZE(4),
        .CASE_ID(1),
        .MAX_CYCLES(160)
    ) case_4x4_4x4 (
        .clk(clk),
        .rst_n(rst_n),
        .case_done(done1),
        .case_pass(pass1)
    );

    SequencerAlignCase #(
        .M_SIZE(8),
        .K_SIZE(8),
        .N_SIZE(8),
        .CASE_ID(2),
        .MAX_CYCLES(300)
    ) case_8x8_8x8 (
        .clk(clk),
        .rst_n(rst_n),
        .case_done(done2),
        .case_pass(pass2)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        tb_cycle = 0;
        run0 = 1'b0;
        run1 = 1'b0;
        run2 = 1'b0;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        $display("");
        $display("========== RUN CASE0: 2x3 * 3x2 ==========");
        case_2x3_3x2.run_case();

        $display("");
        $display("========== RUN CASE1: 4x4 * 4x4 ==========");
        case_4x4_4x4.run_case();

        $display("");
        $display("========== RUN CASE2: 8x8 * 8x8 ==========");
        case_8x8_8x8.run_case();

        if (!(done0 && done1 && done2)) begin
            $display("SEQUENCER_ALIGN_TB TIMEOUT");
        end else if (pass0 && pass1 && pass2) begin
            $display("SEQUENCER_ALIGN_TB PASSED");
        end else begin
            $display("SEQUENCER_ALIGN_TB FAILED");
        end

        #20;
        $finish;
    end

endmodule
