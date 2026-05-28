`timescale 1ns / 1ps

module sequencer_tb;
    localparam DATA_W = 8;
    localparam ACC_W  = 32;
    localparam TB_VERBOSE_TRACE = 1'b0;
    localparam TB_PRINT_MATRICES = 1'b0;

    reg i_clk;
    reg i_rst_n;
    integer cycle;

    reg                      i_size_ld;
    reg [15:0]               i_m_size;
    reg [15:0]               i_k_size;
    reg [15:0]               i_n_size;
    reg [31:0]               i_act_base_addr;
    reg [31:0]               i_wgt_base_addr;

    reg [4*DATA_W-1:0]       i_act_stream_data;
    reg                      i_act_stream_valid;
    wire                     o_act_stream_ready;
    wire [31:0]              o_act_base_addr;
    wire                     o_act_base_valid;

    reg [4*DATA_W-1:0]       i_wgt_stream_data;
    reg                      i_wgt_stream_valid;
    wire                     o_wgt_stream_ready;
    wire [31:0]              o_wgt_base_addr;
    wire                     o_wgt_base_valid;

    wire                     o_array_en;
    wire                     o_array_clear;
    wire [4*DATA_W-1:0]      o_act_array_data;
    wire [3:0]               o_wgt_array_ld;
    wire [4*DATA_W-1:0]      o_wgt_array_data;

    wire [15:0]              o_psum_zero;
    wire [1:0]               o_psum_sel;
    wire [4*ACC_W-1:0]       i_psum_data;
    wire [4*4*ACC_W-1:0]     o_c_data_flat;
    reg [4*DATA_W-1:0]       act_sram [0:255];
    reg [31:0]               act_rd_addr;
    reg [15:0]               act_words_left;
    reg                      act_loader_busy;
    reg [3:0]                act_base_delay_cfg;
    reg [3:0]                act_base_delay_left;
    reg [31:0]               act_pending_base_addr;
    reg                      act_base_pending;
    reg [4*DATA_W-1:0]       wgt_sram [0:255];
    reg [31:0]               wgt_rd_addr;
    reg [15:0]               wgt_words_left;
    reg                      wgt_loader_busy;
    reg [3:0]                wgt_base_delay_cfg;
    reg [3:0]                wgt_base_delay_left;
    reg [31:0]               wgt_pending_base_addr;
    reg                      wgt_base_pending;
    reg [DATA_W-1:0]         ref_act_lane0;
    reg [DATA_W-1:0]         ref_act_lane1 [0:1];
    reg [DATA_W-1:0]         ref_act_lane2 [0:2];
    reg [DATA_W-1:0]         ref_act_lane3 [0:3];
    reg [DATA_W-1:0]         ref_wgt_lane0;
    reg [DATA_W-1:0]         ref_wgt_lane1 [0:1];
    reg [DATA_W-1:0]         ref_wgt_lane2 [0:2];
    reg [DATA_W-1:0]         ref_wgt_lane3 [0:3];
    reg [1:0]                ref_lane_sequen;
    reg [4*DATA_W-1:0]       ref_act_in;
    reg [4*DATA_W-1:0]       ref_wgt_in;
    reg [3:0]                ref_wgt_ld;
    integer                  ref_act_phase;
    integer                  gold_act_phase;
    reg [4*DATA_W-1:0]       gold_act_in;
    integer                  gold_wgt_phase;
    integer                  gold_n_tile;
    reg [4*DATA_W-1:0]       gold_wgt_in;
    reg [3:0]                gold_wgt_ld;
    reg [15:0]               gold_psum_zero;
    integer                  gold_psum_wave_phase [0:3];
    integer                  gold_psum_start_valid;
    integer                  gold_psum_start_group;
    integer                  gold_psum_i;
    integer                  gold_psum_out_phase;
    integer                  gold_psum_tile_cols;
    integer                  gold_psum_tail_delay;
    reg [1:0]                gold_psum_sel;
    reg                      gold_psum_array_en_d;
    integer                  init_i;
    integer                  current_case;
    integer                  wgt_value_scale;

    wire                     tb_act_stream_en;
    wire                     tb_wgt_stream_en;
    wire                     tb_pipe_run;
    wire                     tb_array_feed_en;

    assign tb_act_stream_en = i_act_stream_valid && o_act_stream_ready;
    assign tb_wgt_stream_en = i_wgt_stream_valid && o_wgt_stream_ready;
    assign tb_pipe_run = ((dut.state == 3'd1) && tb_wgt_stream_en) ||
                         ((dut.state == 3'd2) &&
                          tb_act_stream_en &&
                          (tb_wgt_stream_en || dut.wgt_all_read_done)) ||
                         (dut.state == 3'd4);
    assign tb_array_feed_en = dut.array_feed_en;

    function [DATA_W-1:0] ref_a_value_from_stream;
        input integer word_idx;
        input integer lane_idx;
        integer m_blk;
        integer k_idx;
        integer m_idx;
        begin
            if ((word_idx < 0) ||
                (word_idx >= (i_k_size * ((i_m_size + 16'd3) >> 2)))) begin
                ref_a_value_from_stream = {DATA_W{1'b0}};
            end
            else begin
                m_blk = word_idx / i_k_size;
                k_idx = word_idx % i_k_size;
                m_idx = m_blk * 4 + lane_idx;
                if (m_idx < i_m_size)
                    ref_a_value_from_stream = m_idx * i_k_size + k_idx + 1;
                else
                    ref_a_value_from_stream = {DATA_W{1'b0}};
            end
        end
    endfunction

    function [4*DATA_W-1:0] ref_act_array_from_phase;
        input integer phase;
        integer row_idx;
        integer lane_idx;
        integer word_idx;
        begin
            ref_act_array_from_phase = {4*DATA_W{1'b0}};
            for (row_idx = 0; row_idx < 4; row_idx = row_idx + 1) begin
                lane_idx = (phase + 4 - row_idx) % 4;
                word_idx = phase - lane_idx;
                ref_act_array_from_phase[row_idx*DATA_W +: DATA_W] =
                    ref_a_value_from_stream(word_idx, lane_idx);
            end
        end
    endfunction

    function [DATA_W-1:0] ref_b_value;
        input integer k_idx;
        input integer n_idx;
        begin
            if ((k_idx < 0) || (k_idx >= i_k_size) ||
                (n_idx < 0) || (n_idx >= i_n_size))
                ref_b_value = {DATA_W{1'b0}};
            else
                ref_b_value = (k_idx * i_n_size + n_idx + 1) * wgt_value_scale;
        end
    endfunction

    function [4*DATA_W-1:0] ref_wgt_array_from_phase;
        input integer phase;
        input integer n_tile_idx;
        integer col_idx;
        integer k_idx;
        integer n_idx;
        integer word_idx;
        begin
            ref_wgt_array_from_phase = {4*DATA_W{1'b0}};
            for (col_idx = 0; col_idx < 4; col_idx = col_idx + 1) begin
                n_idx = n_tile_idx * 4 + col_idx;
                word_idx = phase + 1 - col_idx;
                if ((word_idx >= 0) &&
                    (word_idx < (i_k_size * ((i_m_size + 16'd3) >> 2))) &&
                    (n_idx < i_n_size)) begin
                    k_idx = word_idx % i_k_size;
                    ref_wgt_array_from_phase[col_idx*DATA_W +: DATA_W] =
                        ref_b_value(k_idx, n_idx);
                end
            end
        end
    endfunction

    function [3:0] ref_psum_zero_mask;
        input integer phase;
        begin
            case (phase)
                1: ref_psum_zero_mask = 4'b0001;
                2: ref_psum_zero_mask = 4'b0011;
                3: ref_psum_zero_mask = 4'b0111;
                4: ref_psum_zero_mask = 4'b1111;
                5: ref_psum_zero_mask = 4'b1110;
                6: ref_psum_zero_mask = 4'b1100;
                7: ref_psum_zero_mask = 4'b1000;
                default: ref_psum_zero_mask = 4'b0000;
            endcase
        end
    endfunction

    Sequencer #(
        .DATA_W(DATA_W),
        .ACC_W (ACC_W),
        .MAX_M_SIZE(4),
        .MAX_K_SIZE(4),
        .MAX_N_SIZE(4)
    ) dut (
        .i_clk              (i_clk),
        .i_rst_n            (i_rst_n),
        .i_size_ld          (i_size_ld),
        .i_m_size           (i_m_size),
        .i_k_size           (i_k_size),
        .i_n_size           (i_n_size),
        .i_act_base_addr (i_act_base_addr),
        .i_wgt_base_addr (i_wgt_base_addr),
        .i_act_stream_data    (i_act_stream_data),
        .i_act_stream_valid (i_act_stream_valid),
        .o_act_stream_ready (o_act_stream_ready),
        .o_act_base_addr    (o_act_base_addr),
        .o_act_base_valid   (o_act_base_valid),
        .i_wgt_stream_data    (i_wgt_stream_data),
        .i_wgt_stream_valid (i_wgt_stream_valid),
        .o_wgt_stream_ready (o_wgt_stream_ready),
        .o_wgt_base_addr    (o_wgt_base_addr),
        .o_wgt_base_valid   (o_wgt_base_valid),
        .o_array_en               (o_array_en),
        .o_array_clear            (o_array_clear),
        .o_act_array_data           (o_act_array_data),
        .o_wgt_array_ld           (o_wgt_array_ld),
        .o_wgt_array_data           (o_wgt_array_data),
        .o_psum_zero        (o_psum_zero),
        .o_psum_sel     (o_psum_sel),
        .i_psum_data         (i_psum_data),
        .o_c_data_flat       (o_c_data_flat)
    );

    SystolicArray4x4 #(
        .DATA_W(DATA_W),
        .ACC_W (ACC_W)
    ) array_u (
        .clk          (i_clk),
        .rst_n        (i_rst_n),
        .en           (o_array_en),
        .wgt_ld       (o_wgt_array_ld),
        .wgt_in       (o_wgt_array_data),
        .clear        (o_array_clear),
        .act_in       (o_act_array_data),
        .psum_zero    (o_psum_zero),
        .psum_out_sel (o_psum_sel),
        .psum_out     (i_psum_data)
    );

    initial begin
        i_clk = 1'b0;
        forever #5 i_clk = ~i_clk;
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            cycle <= 0;
        else
            cycle <= cycle + 1;
    end

    task wait_done_state;
        integer timeout;
        reg saw_done;
        begin
            timeout = 0;
            saw_done = 1'b0;
            while (!saw_done && timeout < 220) begin
                timeout = timeout + 1;
                @(negedge i_clk);
                if (dut.state == 3'd6)
                    saw_done = 1'b1;
            end

            if (!saw_done) begin
                $display("ERROR: case=%0d timeout waiting for S_DONE at t=%0t", current_case, $time);
                $finish;
            end

            $display("DONE: case=%0d reached S_DONE at cycle=%0d", current_case, cycle);
        end
    endtask

    task wait_idle_state;
        integer timeout;
        begin
            timeout = 0;
            while (dut.state != 3'd0 && timeout < 20) begin
                timeout = timeout + 1;
                @(negedge i_clk);
            end

            if (dut.state != 3'd0) begin
                $display("ERROR: case=%0d timeout waiting for S_IDLE after S_DONE at t=%0t", current_case, $time);
                $finish;
            end
        end
    endtask

    task clear_sram;
        begin
            for (init_i = 0; init_i < 256; init_i = init_i + 1) begin
                act_sram[init_i] = {4*DATA_W{1'b0}};
                wgt_sram[init_i] = {4*DATA_W{1'b0}};
            end
        end
    endtask

    task load_stream_case;
        input [15:0] tm;
        input [15:0] tk;
        input [15:0] tn;
        input [31:0] act_base;
        input [31:0] wgt_base;
        integer m_blk;
        integer n_blk;
        integer row;
        integer col;
        integer lane_row;
        integer lane_col;
        reg [DATA_W-1:0] v0;
        reg [DATA_W-1:0] v1;
        reg [DATA_W-1:0] v2;
        reg [DATA_W-1:0] v3;
        begin
            clear_sram();

            for (m_blk = 0; m_blk < ((tm + 3) >> 2); m_blk = m_blk + 1) begin
                for (col = 0; col < tk; col = col + 1) begin
                    lane_row = m_blk * 4;
                    v0 = (lane_row + 0 < tm) ? ((lane_row + 0) * tk + col + 1) : 0;
                    v1 = (lane_row + 1 < tm) ? ((lane_row + 1) * tk + col + 1) : 0;
                    v2 = (lane_row + 2 < tm) ? ((lane_row + 2) * tk + col + 1) : 0;
                    v3 = (lane_row + 3 < tm) ? ((lane_row + 3) * tk + col + 1) : 0;
                    act_sram[act_base + m_blk * tk + col] = {v3, v2, v1, v0};
                end
            end

            for (n_blk = 0; n_blk < ((tn + 3) >> 2); n_blk = n_blk + 1) begin
                for (row = 0; row < tk; row = row + 1) begin
                    lane_col = n_blk * 4;
                    v0 = (lane_col + 0 < tn) ? ((row * tn + lane_col + 1) * wgt_value_scale) : 0;
                    v1 = (lane_col + 1 < tn) ? ((row * tn + lane_col + 2) * wgt_value_scale) : 0;
                    v2 = (lane_col + 2 < tn) ? ((row * tn + lane_col + 3) * wgt_value_scale) : 0;
                    v3 = (lane_col + 3 < tn) ? ((row * tn + lane_col + 4) * wgt_value_scale) : 0;
                    wgt_sram[wgt_base + n_blk * tk + row] = {v3, v2, v1, v0};
                end
            end
        end
    endtask

    task print_matrices;
        input [15:0] tm;
        input [15:0] tk;
        input [15:0] tn;
        integer row;
        integer col;
        begin
            $display("A matrix:");
            for (row = 0; row < tm; row = row + 1) begin
                $write("  ");
                for (col = 0; col < tk; col = col + 1)
                    $write("%0d ", row * tk + col + 1);
                $write("\n");
            end

            $display("B matrix:");
            for (row = 0; row < tk; row = row + 1) begin
                $write("  ");
                for (col = 0; col < tn; col = col + 1)
                    $write("%0d ", (row * tn + col + 1) * wgt_value_scale);
                $write("\n");
            end
        end
    endtask

    task run_case;
        input integer case_id;
        input [15:0] tm;
        input [15:0] tk;
        input [15:0] tn;
        input [31:0] act_base;
        input [31:0] wgt_base;
        input integer wgt_scale;
        begin
            current_case = case_id;
            wgt_value_scale = wgt_scale;
            load_stream_case(tm, tk, tn, act_base, wgt_base);

            i_rst_n = 1'b0;
            i_size_ld = 1'b0;
            i_m_size = 16'd0;
            i_k_size = 16'd0;
            i_n_size = 16'd0;
            i_act_base_addr = act_base;
            i_wgt_base_addr = wgt_base;
            repeat (4) @(negedge i_clk);
            i_rst_n = 1'b1;

            @(negedge i_clk);
            i_m_size = tm;
            i_k_size = tk;
            i_n_size = tn;
            i_size_ld = 1'b1;
            $display("BEGIN: case=%0d A=%0dx%0d B=%0dx%0d wgt_scale=%0d", case_id, tm, tk, tk, tn, wgt_scale);
            if (case_id == 7) begin
                $display("CASE7 TRACE LEGEND:");
                $display("  ctrl: state hold pipe_run array_en");
                $display("  loop: n_tile_idx wgt_replay_idx act_k_idx wgt_k_idx act_m_block_idx wgt_tile_done act_tile_done wgt_replay_done");
                $display("  act : stream_valid/ready act_output_valid act_array_data={row0,row1,row2,row3}");
                $display("  wgt : stream_valid/ready wgt_output_valid wgt_array_data={col0,col1,col2,col3} wgt_ld_phase");
                $display("  psum: psum_zero psum_sel psum_data={col3,col2,col1,col0}");
            end
            if (TB_PRINT_MATRICES)
                print_matrices(tm, tk, tn);
            @(negedge i_clk);
            i_size_ld = 1'b0;

            wait_done_state();
            wait_idle_state();
            repeat (4) @(negedge i_clk);
        end
    endtask

    task run_case_no_reset;
        input integer case_id;
        input [15:0] tm;
        input [15:0] tk;
        input [15:0] tn;
        input [31:0] act_base;
        input [31:0] wgt_base;
        input integer wgt_scale;
        begin
            current_case = case_id;
            wgt_value_scale = wgt_scale;
            load_stream_case(tm, tk, tn, act_base, wgt_base);

            if (dut.state != 3'd0) begin
                $display("ERROR: case=%0d expected S_IDLE before no-reset start, state=%0d", case_id, dut.state);
                $finish;
            end

            @(negedge i_clk);
            i_act_base_addr = act_base;
            i_wgt_base_addr = wgt_base;
            i_m_size = tm;
            i_k_size = tk;
            i_n_size = tn;
            i_size_ld = 1'b1;
            $display("BEGIN: case=%0d A=%0dx%0d B=%0dx%0d wgt_scale=%0d no_reset=1", case_id, tm, tk, tk, tn, wgt_scale);
            if (TB_PRINT_MATRICES)
                print_matrices(tm, tk, tn);
            @(negedge i_clk);
            i_size_ld = 1'b0;

            wait_done_state();
            wait_idle_state();
            repeat (4) @(negedge i_clk);
        end
    endtask

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            i_act_stream_valid <= 1'b0;
            i_act_stream_data <= {4*DATA_W{1'b0}};
            act_rd_addr <= 32'b0;
            act_words_left <= 16'b0;
            act_loader_busy <= 1'b0;
            act_base_delay_left <= 4'b0;
            act_pending_base_addr <= 32'b0;
            act_base_pending <= 1'b0;
        end else if (o_act_base_valid) begin
            if (act_base_delay_cfg == 4'b0) begin
                act_rd_addr <= o_act_base_addr;
                act_words_left <= i_k_size * ((i_m_size + 16'd3) >> 2);
                i_act_stream_data <= act_sram[o_act_base_addr];
                i_act_stream_valid <= 1'b1;
                act_loader_busy <= 1'b1;
            end else begin
                act_pending_base_addr <= o_act_base_addr;
                act_base_delay_left <= act_base_delay_cfg;
                act_base_pending <= 1'b1;
                i_act_stream_valid <= 1'b0;
                i_act_stream_data <= {4*DATA_W{1'b0}};
                act_loader_busy <= 1'b0;
            end
        end else if (act_base_pending) begin
            if (act_base_delay_left == 4'd1) begin
                act_rd_addr <= act_pending_base_addr;
                act_words_left <= i_k_size * ((i_m_size + 16'd3) >> 2);
                i_act_stream_data <= act_sram[act_pending_base_addr];
                i_act_stream_valid <= 1'b1;
                act_loader_busy <= 1'b1;
                act_base_pending <= 1'b0;
                act_base_delay_left <= 4'b0;
            end else begin
                act_base_delay_left <= act_base_delay_left - 1'b1;
            end
        end else if (i_act_stream_valid && o_act_stream_ready) begin
            if (act_words_left == 16'd1) begin
                i_act_stream_valid <= 1'b0;
                i_act_stream_data <= {4*DATA_W{1'b0}};
                act_words_left <= 16'b0;
                act_loader_busy <= 1'b0;
            end else begin
                act_rd_addr <= act_rd_addr + 1'b1;
                act_words_left <= act_words_left - 1'b1;
                i_act_stream_data <= act_sram[act_rd_addr + 1'b1];
            end
        end
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            i_wgt_stream_valid <= 1'b0;
            i_wgt_stream_data <= {4*DATA_W{1'b0}};
            wgt_rd_addr <= 32'b0;
            wgt_words_left <= 16'b0;
            wgt_loader_busy <= 1'b0;
            wgt_base_delay_left <= 4'b0;
            wgt_pending_base_addr <= 32'b0;
            wgt_base_pending <= 1'b0;
        end else if (o_wgt_base_valid) begin
            if (wgt_base_delay_cfg == 4'b0) begin
                wgt_rd_addr <= o_wgt_base_addr;
                wgt_words_left <= i_k_size;
                i_wgt_stream_data <= wgt_sram[o_wgt_base_addr];
                i_wgt_stream_valid <= 1'b1;
                wgt_loader_busy <= 1'b1;
            end else begin
                wgt_pending_base_addr <= o_wgt_base_addr;
                wgt_base_delay_left <= wgt_base_delay_cfg;
                wgt_base_pending <= 1'b1;
                i_wgt_stream_valid <= 1'b0;
                i_wgt_stream_data <= {4*DATA_W{1'b0}};
                wgt_loader_busy <= 1'b0;
            end
        end else if (wgt_base_pending) begin
            if (wgt_base_delay_left == 4'd1) begin
                wgt_rd_addr <= wgt_pending_base_addr;
                wgt_words_left <= i_k_size;
                i_wgt_stream_data <= wgt_sram[wgt_pending_base_addr];
                i_wgt_stream_valid <= 1'b1;
                wgt_loader_busy <= 1'b1;
                wgt_base_pending <= 1'b0;
                wgt_base_delay_left <= 4'b0;
            end else begin
                wgt_base_delay_left <= wgt_base_delay_left - 1'b1;
            end
        end else if (i_wgt_stream_valid && o_wgt_stream_ready) begin
            if (wgt_words_left == 16'd1) begin
                i_wgt_stream_valid <= 1'b0;
                i_wgt_stream_data <= {4*DATA_W{1'b0}};
                wgt_words_left <= 16'b0;
                wgt_loader_busy <= 1'b0;
            end else begin
                wgt_rd_addr <= wgt_rd_addr + 1'b1;
                wgt_words_left <= wgt_words_left - 1'b1;
                i_wgt_stream_data <= wgt_sram[wgt_rd_addr + 1'b1];
            end
        end
    end

    initial begin
        i_rst_n = 1'b0;
        cycle = 0;
        i_size_ld = 1'b0;
        i_m_size = 16'd0;
        i_k_size = 16'd0;
        i_n_size = 16'd0;
        i_act_base_addr = 32'd100;
        i_wgt_base_addr = 32'd200;
        current_case = 0;
        wgt_value_scale = 1;
        act_base_delay_cfg = 4'd0;
        wgt_base_delay_cfg = 4'd0;

        run_case(1, 16'd5, 16'd6, 16'd8, 32'd100, 32'd200, 1);
        run_case(2, 16'd2, 16'd3, 16'd2, 32'd100, 32'd200, 1);
        run_case(3, 16'd8, 16'd9, 16'd6, 32'd100, 32'd200, 1);
        wgt_base_delay_cfg = 4'd2;
        run_case(4, 16'd5, 16'd6, 16'd8, 32'd100, 32'd200, 1);
        wgt_base_delay_cfg = 4'd0;
        act_base_delay_cfg = 4'd2;
        run_case(5, 16'd5, 16'd6, 16'd8, 32'd100, 32'd200, 1);
        act_base_delay_cfg = 4'd0;
        run_case_no_reset(6, 16'd2, 16'd3, 16'd2, 32'd100, 32'd200, 1);
        run_case(7, 16'd5, 16'd3, 16'd2, 32'd100, 32'd200, 10);

        $finish;
    end

    task dump_cycle_info;
        begin
            $display(
                "cycle=%0d state=%0d hold=%b aout=%b wout=%b o_array_en=%b | act v/r=%b/%b base=%b:%0d stream=%0d,%0d,%0d,%0d o_act_array_data=%0d,%0d,%0d,%0d ref=%0d,%0d,%0d,%0d",
                cycle,
                dut.state,
                dut.hold,
                dut.act_out_en,
                dut.wgt_out_en,
                o_array_en,
                i_act_stream_valid,
                o_act_stream_ready,
                o_act_base_valid,
                o_act_base_addr,
                i_act_stream_data[7:0],
                i_act_stream_data[15:8],
                i_act_stream_data[23:16],
                i_act_stream_data[31:24],
                o_act_array_data[31:24],
                o_act_array_data[23:16],
                o_act_array_data[15:8],
                o_act_array_data[7:0],
                ref_act_in[31:24],
                ref_act_in[23:16],
                ref_act_in[15:8],
                ref_act_in[7:0]
            );
            $display(
                "  wgt v/r=%b/%b base=%b:%0d stream=%0d,%0d,%0d,%0d o_wgt_array_data=%0d,%0d,%0d,%0d ref=%0d,%0d,%0d,%0d ld=%b exp_ld=%b",
                i_wgt_stream_valid,
                o_wgt_stream_ready,
                o_wgt_base_valid,
                o_wgt_base_addr,
                i_wgt_stream_data[7:0],
                i_wgt_stream_data[15:8],
                i_wgt_stream_data[23:16],
                i_wgt_stream_data[31:24],
                o_wgt_array_data[31:24],
                o_wgt_array_data[23:16],
                o_wgt_array_data[15:8],
                o_wgt_array_data[7:0],
                ref_wgt_in[31:24],
                ref_wgt_in[23:16],
                ref_wgt_in[15:8],
                ref_wgt_in[7:0],
                o_wgt_array_ld,
                gold_wgt_ld
            );
        end
    endtask

    task dump_array_cycle_info;
        begin
            $display(
                "case=%0d cycle=%0d state=%0d array_en=%b | o_act_array_data(row0,row1,row2,row3)={%0d,%0d,%0d,%0d} | o_wgt_array_data(col0,col1,col2,col3)={%0d,%0d,%0d,%0d} wgt_ld=%b psum_zero=%h exp_psum_zero=%h",
                current_case,
                cycle,
                dut.state,
                o_array_en,
                o_act_array_data[7:0],
                o_act_array_data[15:8],
                o_act_array_data[23:16],
                o_act_array_data[31:24],
                o_wgt_array_data[7:0],
                o_wgt_array_data[15:8],
                o_wgt_array_data[23:16],
                o_wgt_array_data[31:24],
                o_wgt_array_ld,
                o_psum_zero,
                o_array_en ? gold_psum_zero : 16'hxxxx
            );
        end
    endtask

    task dump_psum_cycle_info;
        begin
            $display(
                "PSUM case=%0d cycle=%0d array_en=%b psum_zero=%h psum_sel=%0d psum_out(c0,c1,c2,c3)={%0d,%0d,%0d,%0d}",
                current_case,
                cycle,
                o_array_en,
                o_psum_zero,
                o_psum_sel,
                $signed(i_psum_data[ACC_W-1:0]),
                $signed(i_psum_data[2*ACC_W-1:ACC_W]),
                $signed(i_psum_data[3*ACC_W-1:2*ACC_W]),
                $signed(i_psum_data[4*ACC_W-1:3*ACC_W])
            );
            if (current_case == 7) begin
                $display(
                    "  rows r0={%0d,%0d,%0d,%0d} r1={%0d,%0d,%0d,%0d} r2={%0d,%0d,%0d,%0d} r3={%0d,%0d,%0d,%0d}",
                    $signed(array_u.psum_v[1][0]), $signed(array_u.psum_v[1][1]), $signed(array_u.psum_v[1][2]), $signed(array_u.psum_v[1][3]),
                    $signed(array_u.psum_v[2][0]), $signed(array_u.psum_v[2][1]), $signed(array_u.psum_v[2][2]), $signed(array_u.psum_v[2][3]),
                    $signed(array_u.psum_v[3][0]), $signed(array_u.psum_v[3][1]), $signed(array_u.psum_v[3][2]), $signed(array_u.psum_v[3][3]),
                    $signed(array_u.psum_v[4][0]), $signed(array_u.psum_v[4][1]), $signed(array_u.psum_v[4][2]), $signed(array_u.psum_v[4][3])
                );
            end
        end
    endtask

    always @(posedge i_clk) begin
        if (i_rst_n && (dut.state == 3'd2) &&
            i_act_stream_valid && o_act_stream_ready &&
            !(i_wgt_stream_valid && o_wgt_stream_ready) && !dut.wgt_all_read_done) begin
            $display("ERROR: case=%0d cycle=%0d act advanced while waiting for wgt stream", current_case, cycle);
            dump_cycle_info();
            $finish;
        end
    end

    always @(posedge i_clk) begin
        if (i_rst_n && (dut.state == 3'd2) &&
            i_wgt_stream_valid && o_wgt_stream_ready &&
            !(i_act_stream_valid && o_act_stream_ready) && !dut.wgt_all_read_done) begin
            $display("ERROR: case=%0d cycle=%0d wgt advanced while waiting for act stream", current_case, cycle);
            dump_cycle_info();
            $finish;
        end
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            ref_act_lane0 <= 'b0;
            ref_act_lane1[0] <= 'b0;
            ref_act_lane1[1] <= 'b0;
            ref_act_lane2[0] <= 'b0;
            ref_act_lane2[1] <= 'b0;
            ref_act_lane2[2] <= 'b0;
            ref_act_lane3[0] <= 'b0;
            ref_act_lane3[1] <= 'b0;
            ref_act_lane3[2] <= 'b0;
            ref_act_lane3[3] <= 'b0;
            ref_wgt_lane0 <= 'b0;
            ref_wgt_lane1[0] <= 'b0;
            ref_wgt_lane1[1] <= 'b0;
            ref_wgt_lane2[0] <= 'b0;
            ref_wgt_lane2[1] <= 'b0;
            ref_wgt_lane2[2] <= 'b0;
            ref_wgt_lane3[0] <= 'b0;
            ref_wgt_lane3[1] <= 'b0;
            ref_wgt_lane3[2] <= 'b0;
            ref_wgt_lane3[3] <= 'b0;
            ref_lane_sequen <= 2'b0;
            ref_act_in <= 'b0;
            ref_wgt_in <= 'b0;
            ref_wgt_ld <= 4'b0001;
            ref_act_phase <= -1;
        end
        else if (i_size_ld && (dut.state == 3'd0)) begin
            ref_act_lane0 <= 'b0;
            ref_act_lane1[0] <= 'b0;
            ref_act_lane1[1] <= 'b0;
            ref_act_lane2[0] <= 'b0;
            ref_act_lane2[1] <= 'b0;
            ref_act_lane2[2] <= 'b0;
            ref_act_lane3[0] <= 'b0;
            ref_act_lane3[1] <= 'b0;
            ref_act_lane3[2] <= 'b0;
            ref_act_lane3[3] <= 'b0;
            ref_wgt_lane0 <= 'b0;
            ref_wgt_lane1[0] <= 'b0;
            ref_wgt_lane1[1] <= 'b0;
            ref_wgt_lane2[0] <= 'b0;
            ref_wgt_lane2[1] <= 'b0;
            ref_wgt_lane2[2] <= 'b0;
            ref_wgt_lane3[0] <= 'b0;
            ref_wgt_lane3[1] <= 'b0;
            ref_wgt_lane3[2] <= 'b0;
            ref_wgt_lane3[3] <= 'b0;
            ref_lane_sequen <= 2'b0;
            ref_act_in <= 'b0;
            ref_wgt_in <= 'b0;
            ref_wgt_ld <= 4'b0001;
            ref_act_phase <= -1;
        end
        else begin
            if (tb_pipe_run) begin
                if (tb_act_stream_en) begin
                    ref_act_lane0 <= i_act_stream_data[DATA_W-1:0];
                    ref_act_lane1[0] <= ref_act_lane1[1];
                    ref_act_lane1[1] <= i_act_stream_data[2*DATA_W-1:DATA_W];
                    ref_act_lane2[0] <= ref_act_lane2[1];
                    ref_act_lane2[1] <= ref_act_lane2[2];
                    ref_act_lane2[2] <= i_act_stream_data[3*DATA_W-1:2*DATA_W];
                    ref_act_lane3[0] <= ref_act_lane3[1];
                    ref_act_lane3[1] <= ref_act_lane3[2];
                    ref_act_lane3[2] <= ref_act_lane3[3];
                    ref_act_lane3[3] <= i_act_stream_data[4*DATA_W-1:3*DATA_W];
                end
                else begin
                    ref_act_lane0 <= {DATA_W{1'b0}};
                    ref_act_lane1[0] <= ref_act_lane1[1];
                    ref_act_lane1[1] <= {DATA_W{1'b0}};
                    ref_act_lane2[0] <= ref_act_lane2[1];
                    ref_act_lane2[1] <= ref_act_lane2[2];
                    ref_act_lane2[2] <= {DATA_W{1'b0}};
                    ref_act_lane3[0] <= ref_act_lane3[1];
                    ref_act_lane3[1] <= ref_act_lane3[2];
                    ref_act_lane3[2] <= ref_act_lane3[3];
                    ref_act_lane3[3] <= {DATA_W{1'b0}};
                end

                if (tb_wgt_stream_en) begin
                    ref_wgt_lane0 <= i_wgt_stream_data[DATA_W-1:0];
                    ref_wgt_lane1[0] <= ref_wgt_lane1[1];
                    ref_wgt_lane1[1] <= i_wgt_stream_data[2*DATA_W-1:DATA_W];
                    ref_wgt_lane2[0] <= ref_wgt_lane2[1];
                    ref_wgt_lane2[1] <= ref_wgt_lane2[2];
                    ref_wgt_lane2[2] <= i_wgt_stream_data[3*DATA_W-1:2*DATA_W];
                    ref_wgt_lane3[0] <= ref_wgt_lane3[1];
                    ref_wgt_lane3[1] <= ref_wgt_lane3[2];
                    ref_wgt_lane3[2] <= ref_wgt_lane3[3];
                    ref_wgt_lane3[3] <= i_wgt_stream_data[4*DATA_W-1:3*DATA_W];
                end
                else if (dut.wgt_group_cnt == 14'b0) begin
                    ref_wgt_lane0 <= {DATA_W{1'b0}};
                    ref_wgt_lane1[0] <= ref_wgt_lane1[1];
                    ref_wgt_lane1[1] <= {DATA_W{1'b0}};
                    ref_wgt_lane2[0] <= ref_wgt_lane2[1];
                    ref_wgt_lane2[1] <= ref_wgt_lane2[2];
                    ref_wgt_lane2[2] <= {DATA_W{1'b0}};
                    ref_wgt_lane3[0] <= ref_wgt_lane3[1];
                    ref_wgt_lane3[1] <= ref_wgt_lane3[2];
                    ref_wgt_lane3[2] <= ref_wgt_lane3[3];
                    ref_wgt_lane3[3] <= {DATA_W{1'b0}};
                end

                ref_wgt_in <= {ref_wgt_lane3[0], ref_wgt_lane2[0], ref_wgt_lane1[0], ref_wgt_lane0};
                ref_wgt_ld <= {ref_wgt_ld[2:0], ref_wgt_ld[3]};
            end

            if (o_act_base_valid) begin
                ref_act_phase <= -1;
                ref_act_in <= 'b0;
            end
            else if (o_array_en) begin
                if (ref_act_phase < 0) begin
                    ref_act_in <= 'b0;
                    ref_act_phase <= 0;
                end
                else begin
                    ref_act_in <= ref_act_array_from_phase(ref_act_phase);
                    ref_act_phase <= ref_act_phase + 1;
                end
            end
        end
    end

    always @(negedge i_clk) begin
        if (!i_rst_n || i_size_ld) begin
            gold_act_phase = -1;
            gold_act_in = {4*DATA_W{1'b0}};
            gold_wgt_phase = -1;
            gold_n_tile = 0;
            gold_wgt_in = {4*DATA_W{1'b0}};
            gold_wgt_ld = 4'b0001;
            gold_psum_zero = 16'h0000;
            for (gold_psum_i = 0; gold_psum_i < 4; gold_psum_i = gold_psum_i + 1)
                gold_psum_wave_phase[gold_psum_i] = 0;
            gold_psum_start_valid = 0;
            gold_psum_start_group = 0;
            gold_psum_out_phase = 0;
            gold_psum_tile_cols = 0;
            gold_psum_tail_delay = 0;
            gold_psum_sel = 2'b00;
            gold_psum_array_en_d = 1'b0;
        end
        else begin
            if (o_act_base_valid) begin
                gold_act_phase = -1;
                gold_act_in = {4*DATA_W{1'b0}};
                gold_psum_zero = 16'h0000;
                for (gold_psum_i = 0; gold_psum_i < 4; gold_psum_i = gold_psum_i + 1)
                    gold_psum_wave_phase[gold_psum_i] = 0;
                gold_psum_out_phase = 0;
                gold_psum_array_en_d = 1'b0;
            end
            if (o_wgt_base_valid) begin
                if ((i_k_size != 0) &&
                    (((o_wgt_base_addr - i_wgt_base_addr) / i_k_size) != gold_n_tile)) begin
                    gold_n_tile = (o_wgt_base_addr - i_wgt_base_addr) / i_k_size;
                    gold_wgt_phase = -1;
                    gold_wgt_in = {4*DATA_W{1'b0}};
                    gold_wgt_ld = 4'b0001;
                end
            end

            if ((gold_n_tile * 4 + 4) <= i_n_size)
                gold_psum_tile_cols = 4;
            else
                gold_psum_tile_cols = i_n_size - gold_n_tile * 4;
            if (i_k_size > 4)
                gold_psum_tail_delay =
                    (((i_m_size + 3) >> 2) - 1) * (i_k_size - 4) +
                    ((4 - (((((i_m_size + 3) >> 2) - 1) * i_k_size) % 4)) % 4);
            else
                gold_psum_tail_delay = 0;

            if (o_array_en) begin
                gold_psum_start_valid = 0;
                gold_psum_start_group = 0;
                if ((gold_act_phase >= 0) &&
                    (i_k_size != 0) &&
                    ((gold_act_phase % i_k_size) == 0) &&
                    ((gold_act_phase / i_k_size) < ((i_m_size + 16'd3) >> 2))) begin
                    gold_psum_start_valid = 1;
                    gold_psum_start_group = ((gold_act_phase / i_k_size) * i_k_size) % 4;
                end

                for (gold_psum_i = 0; gold_psum_i < 4; gold_psum_i = gold_psum_i + 1) begin
                    if (gold_psum_start_valid && (gold_psum_start_group == gold_psum_i))
                        gold_psum_wave_phase[gold_psum_i] = 1;
                    else if (gold_psum_wave_phase[gold_psum_i] != 0)
                        gold_psum_wave_phase[gold_psum_i] = gold_psum_wave_phase[gold_psum_i] + 1;

                    if (gold_psum_wave_phase[gold_psum_i] >= 8)
                        gold_psum_wave_phase[gold_psum_i] = 0;
                end

                gold_psum_zero = {
                    ref_psum_zero_mask(gold_psum_wave_phase[3]),
                    ref_psum_zero_mask(gold_psum_wave_phase[2]),
                    ref_psum_zero_mask(gold_psum_wave_phase[1]),
                    ref_psum_zero_mask(gold_psum_wave_phase[0])
                };

                if (tb_array_feed_en) begin
                    if (gold_act_phase < 0)
                        gold_act_in = {4*DATA_W{1'b0}};
                    else
                        gold_act_in = ref_act_array_from_phase(gold_act_phase);

                    if (o_act_array_data !== gold_act_in) begin
                    $display("ERROR: case=%0d cycle=%0d o_act_array_data mismatch got=%0d,%0d,%0d,%0d exp=%0d,%0d,%0d,%0d",
                             current_case, cycle,
                             o_act_array_data[31:24], o_act_array_data[23:16], o_act_array_data[15:8], o_act_array_data[7:0],
                             gold_act_in[31:24], gold_act_in[23:16], gold_act_in[15:8], gold_act_in[7:0]);
                    dump_cycle_info();
                    $finish;
                    end

                    gold_act_phase = gold_act_phase + 1;

                    gold_wgt_in = ref_wgt_array_from_phase(gold_wgt_phase, gold_n_tile);
                    if (o_wgt_array_data !== gold_wgt_in) begin
                        $display("ERROR: case=%0d cycle=%0d o_wgt_array_data mismatch got(col0..3)=%0d,%0d,%0d,%0d exp(col0..3)=%0d,%0d,%0d,%0d phase=%0d n_tile=%0d",
                                 current_case, cycle,
                                 o_wgt_array_data[7:0], o_wgt_array_data[15:8], o_wgt_array_data[23:16], o_wgt_array_data[31:24],
                                 gold_wgt_in[7:0], gold_wgt_in[15:8], gold_wgt_in[23:16], gold_wgt_in[31:24],
                                 gold_wgt_phase, gold_n_tile);
                        dump_cycle_info();
                        $finish;
                    end

                    if (o_wgt_array_ld !== gold_wgt_ld) begin
                        $display("ERROR: case=%0d cycle=%0d o_wgt_array_ld mismatch got=%b exp=%b phase=%0d n_tile=%0d",
                                 current_case, cycle, o_wgt_array_ld, gold_wgt_ld,
                                 gold_wgt_phase, gold_n_tile);
                        dump_cycle_info();
                        $finish;
                    end

                    gold_wgt_phase = gold_wgt_phase + 1;
                    gold_wgt_ld = {gold_wgt_ld[2:0], gold_wgt_ld[3]};
                end

                if (o_psum_zero !== gold_psum_zero) begin
                    $display("ERROR: case=%0d cycle=%0d o_psum_zero mismatch got=%h exp=%h act_phase=%0d start_valid=%0d start_group=%0d",
                             current_case, cycle, o_psum_zero, gold_psum_zero,
                             gold_act_phase, gold_psum_start_valid, gold_psum_start_group);
                    dump_cycle_info();
                    $finish;
                end
            end

            if (gold_psum_array_en_d)
                gold_psum_out_phase = gold_psum_out_phase + 1;
            gold_psum_array_en_d = o_array_en;
        end
    end

    always @(posedge i_clk) begin
        if (TB_VERBOSE_TRACE && i_rst_n &&
            (i_act_stream_valid || i_wgt_stream_valid || o_array_en ||
                      o_act_base_valid || o_wgt_base_valid)) begin
            dump_cycle_info();
        end
    end

    always @(posedge i_clk) begin
        if (i_rst_n && !i_size_ld && (current_case == 7) &&
            ((dut.state != 3'd0) || o_array_en || o_act_base_valid ||
             o_wgt_base_valid || i_act_stream_valid || i_wgt_stream_valid))
            dump_array_cycle_info();
    end

    always @(posedge i_clk) begin
        if (i_rst_n && !i_size_ld && (current_case != 0) &&
            ((dut.state != 3'd0) || o_array_en))
            dump_psum_cycle_info();
    end
endmodule
