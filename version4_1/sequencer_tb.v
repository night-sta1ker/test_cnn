`timescale 1ns / 1ps

module sequencer_tb;
    localparam DATA_W = 8;
    localparam ACC_W  = 32;
    localparam RESULT_SLOTS = 12;
    localparam SRAM_DEPTH = 2048;
    localparam RESULT_MAX_M = 128;
    localparam RESULT_MAX_N = 16;
    localparam RESULT_FIFO_DEPTH = RESULT_MAX_M * RESULT_MAX_N;

`ifdef FSDB
    reg [1023:0] fsdb_file;

    initial begin
        if (!$value$plusargs("fsdbfile=%s", fsdb_file))
            fsdb_file = "wave.fsdb";
        $fsdbDumpfile(fsdb_file);
        $fsdbDumpvars(0, sequencer_tb);
        $fsdbDumpMDA();
    end
`endif

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
    wire                     i_act_stream_valid_dut;
    wire                     o_act_stream_ready;
    wire [31:0]              o_act_base_addr;
    wire                     o_act_base_valid;

    reg [4*DATA_W-1:0]       i_wgt_stream_data;
    reg                      i_wgt_stream_valid;
    wire                     i_wgt_stream_valid_dut;
    wire                     o_wgt_stream_ready;
    wire [31:0]              o_wgt_base_addr;
    wire                     o_wgt_base_valid;

    wire                     o_array_en;
    wire                     o_array_clear;
    wire [4*DATA_W-1:0]      o_act_array_data;
    wire [3:0]               o_wgt_array_ld;
    wire [4*DATA_W-1:0]      o_wgt_array_data;

    wire [15:0]              o_psum_zero;
    wire [7:0]               o_psum_sel;
    wire [3:0]               o_psum_valid;
    wire [4*ACC_W-1:0]       i_psum_data;

    // A new base request cancels any offer from the preceding transaction.
    // The loader presents the first word on the following cycle.
    assign i_act_stream_valid_dut = i_act_stream_valid && !o_act_base_valid;
    assign i_wgt_stream_valid_dut = i_wgt_stream_valid && !o_wgt_base_valid;

    reg [4*DATA_W-1:0]       act_sram [0:SRAM_DEPTH-1];
    reg [31:0]               act_rd_addr;
    reg [15:0]               act_words_left;
    reg                      act_loader_busy;
    reg [3:0]                act_base_delay_cfg;
    reg [3:0]                act_base_delay_left;
    reg [31:0]               act_pending_base_addr;
    reg                      act_base_pending;

    reg [4*DATA_W-1:0]       wgt_sram [0:SRAM_DEPTH-1];
    reg [31:0]               wgt_rd_addr;
    reg [15:0]               wgt_words_left;
    reg                      wgt_loader_busy;
    reg [3:0]                wgt_base_delay_cfg;
    reg [3:0]                wgt_base_delay_left;
    reg [31:0]               wgt_pending_base_addr;
    reg                      wgt_base_pending;

    integer                  current_case;
    integer                  wgt_value_scale;
    integer                  init_i;
    integer                  pass_count;
    integer                  fail_count;
    reg                      case_timed_out;
    integer                  case_id_next;
    integer                  loop_m;
    integer                  loop_n;
    integer                  loop_k;
    reg                      random_data_mode;
    integer                  random_seed;
    integer                  golden_fifo [0:RESULT_FIFO_DEPTH-1];
    integer                  golden_fifo_row [0:RESULT_FIFO_DEPTH-1];
    integer                  golden_fifo_col [0:RESULT_FIFO_DEPTH-1];
    integer                  golden_fifo_count;
    integer                  result_capture_count;

    integer                  result_mem [0:RESULT_MAX_M-1][0:RESULT_MAX_N-1];
    reg                      result_seen [0:RESULT_MAX_M-1][0:RESULT_MAX_N-1];
    integer                  result_duplicate_count;
    integer                  result_capture_error_count;
    integer                  array_en_cycles;
    integer                  case_cycles;
    reg                      case_measure_en;
    reg                      result_first_error_reported;
    reg                      result_active [0:RESULT_SLOTS-1];
    integer                  result_delay [0:RESULT_SLOTS-1];
    integer                  result_phase [0:RESULT_SLOTS-1];
    integer                  result_capture_phase [0:RESULT_SLOTS-1];
    integer                  result_m_block [0:RESULT_SLOTS-1];
    integer                  result_n_tile [0:RESULT_SLOTS-1];
    reg [3:0]                result_slot_mask [0:RESULT_SLOTS-1];
    reg [3:0]                result_lane_used;
    reg [3:0]                zero_start_seen;
    integer                  result_i;
    integer                  result_j;
    integer                  result_alloc;
    integer                  zero_group;

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
        .i_act_base_addr    (i_act_base_addr),
        .i_wgt_base_addr    (i_wgt_base_addr),
        .i_act_stream_data  (i_act_stream_data),
        .i_act_stream_valid (i_act_stream_valid_dut),
        .o_act_stream_ready (o_act_stream_ready),
        .o_act_base_addr    (o_act_base_addr),
        .o_act_base_valid   (o_act_base_valid),
        .i_wgt_stream_data  (i_wgt_stream_data),
        .i_wgt_stream_valid (i_wgt_stream_valid_dut),
        .o_wgt_stream_ready (o_wgt_stream_ready),
        .o_wgt_base_addr    (o_wgt_base_addr),
        .o_wgt_base_valid   (o_wgt_base_valid),
        .o_array_en         (o_array_en),
        .o_array_clear      (o_array_clear),
        .o_act_array_data   (o_act_array_data),
        .o_wgt_array_ld     (o_wgt_array_ld),
        .o_wgt_array_data   (o_wgt_array_data),
        .o_psum_zero        (o_psum_zero),
        .o_psum_sel         (o_psum_sel),
        .i_psum_data        (i_psum_data),
        .o_psum_valid       (o_psum_valid)
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

    function integer random_value;
        input integer row;
        input integer col;
        input integer salt;
        begin
            random_value = (row * 73 + col * 151 + salt * 199 + row * col * 17) % 3;
        end
    endfunction

    function integer act_matrix_value;
        input integer row;
        input integer col;
        input integer tk;
        begin
            if (random_data_mode)
                act_matrix_value = (col == 0) ? 1 : random_value(row, col, random_seed);
            else
                act_matrix_value = row * tk + col + 1;
        end
    endfunction

    function integer wgt_matrix_value;
        input integer row;
        input integer col;
        input integer tn;
        input integer wgt_scale;
        begin
            if (random_data_mode)
                wgt_matrix_value = (row == 0) ? 1 : random_value(row, col, random_seed + 97);
            else
                wgt_matrix_value = (row * tn + col + 1) * wgt_scale;
        end
    endfunction

    function integer golden_c_value;
        input integer row;
        input integer col;
        input integer tk;
        input integer tn;
        input integer wgt_scale;
        integer k_idx;
        integer a_val;
        integer b_val;
        begin
            golden_c_value = 0;
            for (k_idx = 0; k_idx < tk; k_idx = k_idx + 1) begin
                a_val = act_matrix_value(row, k_idx, tk);
                b_val = wgt_matrix_value(k_idx, col, tn, wgt_scale);
                golden_c_value = golden_c_value + a_val * b_val;
            end
        end
    endfunction

    function integer result_tile_rows;
        input integer m_block;
        integer row_base;
        integer rows_left;
        begin
            row_base = m_block * 4;
            rows_left = i_m_size - row_base;
            if (rows_left >= 4)
                result_tile_rows = 4;
            else if (rows_left > 0)
                result_tile_rows = rows_left;
            else
                result_tile_rows = 0;
        end
    endfunction

    function integer result_tile_cols;
        input integer n_tile;
        integer col_base;
        integer cols_left;
        begin
            col_base = n_tile * 4;
            cols_left = i_n_size - col_base;
            if (cols_left >= 4)
                result_tile_cols = 4;
            else if (cols_left > 0)
                result_tile_cols = cols_left;
            else
                result_tile_cols = 0;
        end
    endfunction

    function [3:0] result_valid_from_phase;
        input integer phase;
        input integer rows;
        input integer cols;
        integer lane;
        integer row_in_tile;
        begin
            result_valid_from_phase = 4'b0000;
            for (lane = 0; lane < 4; lane = lane + 1) begin
                row_in_tile = phase - lane;
                if ((lane < cols) && (row_in_tile >= 0) && (row_in_tile < rows))
                    result_valid_from_phase[lane] = 1'b1;
            end
        end
    endfunction

    function integer psum_lane_value;
        input integer lane;
        begin
            case (lane)
                0: psum_lane_value = $signed(i_psum_data[ACC_W-1:0]);
                1: psum_lane_value = $signed(i_psum_data[2*ACC_W-1:ACC_W]);
                2: psum_lane_value = $signed(i_psum_data[3*ACC_W-1:2*ACC_W]);
                3: psum_lane_value = $signed(i_psum_data[4*ACC_W-1:3*ACC_W]);
                default: psum_lane_value = 0;
            endcase
        end
    endfunction

    function integer result_start_delay;
        input integer m_block;
        integer start_phase;
        begin
            start_phase = (m_block * i_k_size) % 4;
            case (start_phase)
                0: result_start_delay = i_k_size;
                1: result_start_delay = i_k_size + 3;
                2: result_start_delay = i_k_size + 2;
                default: result_start_delay = i_k_size + 1;
            endcase
        end
    endfunction

    task wait_done_state;
        integer timeout;
        integer timeout_limit;
        reg saw_done;
        begin
            timeout = 0;
            timeout_limit = i_m_size * i_k_size * ((i_n_size + 3) >> 2) + 2048;
            saw_done = 1'b0;
            while (!saw_done && timeout < timeout_limit) begin
                timeout = timeout + 1;
                @(negedge i_clk);
                if (dut.state == 3'd6)
                    saw_done = 1'b1;
            end

            if (!saw_done) begin
                $display("ERROR: case=%0d timeout state=%0d en=%b act_out=%b wgt_out=%b at t=%0t",
                         current_case, dut.state, o_array_en,
                         dut.act_out_en, dut.wgt_out_en, $time);
                case_timed_out = 1'b1;
            end

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
            for (init_i = 0; init_i < SRAM_DEPTH; init_i = init_i + 1) begin
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
                    v0 = (lane_row + 0 < tm) ? act_matrix_value(lane_row + 0, col, tk) : 0;
                    v1 = (lane_row + 1 < tm) ? act_matrix_value(lane_row + 1, col, tk) : 0;
                    v2 = (lane_row + 2 < tm) ? act_matrix_value(lane_row + 2, col, tk) : 0;
                    v3 = (lane_row + 3 < tm) ? act_matrix_value(lane_row + 3, col, tk) : 0;
                    act_sram[act_base + m_blk * tk + col] = {v3, v2, v1, v0};
                end
            end

            for (n_blk = 0; n_blk < ((tn + 3) >> 2); n_blk = n_blk + 1) begin
                for (row = 0; row < tk; row = row + 1) begin
                    lane_col = n_blk * 4;
                    v0 = (lane_col + 0 < tn) ? wgt_matrix_value(row, lane_col + 0, tn, wgt_value_scale) : 0;
                    v1 = (lane_col + 1 < tn) ? wgt_matrix_value(row, lane_col + 1, tn, wgt_value_scale) : 0;
                    v2 = (lane_col + 2 < tn) ? wgt_matrix_value(row, lane_col + 2, tn, wgt_value_scale) : 0;
                    v3 = (lane_col + 3 < tn) ? wgt_matrix_value(row, lane_col + 3, tn, wgt_value_scale) : 0;
                    wgt_sram[wgt_base + n_blk * tk + row] = {v3, v2, v1, v0};
                end
            end
        end
    endtask

    task clear_result_capture;
        begin
            result_duplicate_count = 0;
            result_capture_error_count = 0;
            result_capture_count = 0;
            result_first_error_reported = 1'b0;
            for (result_i = 0; result_i < RESULT_MAX_M; result_i = result_i + 1) begin
                for (result_j = 0; result_j < RESULT_MAX_N; result_j = result_j + 1) begin
                    result_mem[result_i][result_j] = 0;
                    result_seen[result_i][result_j] = 1'b0;
                end
            end
            for (result_i = 0; result_i < RESULT_SLOTS; result_i = result_i + 1) begin
                result_active[result_i] = 1'b0;
                result_delay[result_i] = 0;
                result_phase[result_i] = 0;
                result_capture_phase[result_i] = 0;
                result_m_block[result_i] = 0;
                result_n_tile[result_i] = 0;
                result_slot_mask[result_i] = 4'b0;
            end
            zero_start_seen = 4'b0;
        end
    endtask

    task build_golden_fifo;
        input [15:0] tm;
        input [15:0] tk;
        input [15:0] tn;
        input integer wgt_scale;
        integer m_block;
        integer n_tile;
        integer rows;
        integer cols;
        integer time_slot;
        integer phase;
        integer lane;
        integer row_in_tile;
        integer row;
        integer col;
        begin
            golden_fifo_count = 0;
            for (n_tile = 0; n_tile < ((tn + 3) >> 2); n_tile = n_tile + 1) begin
                cols = (tn - n_tile * 4 >= 4) ? 4 : tn - n_tile * 4;
                // Each M block launches a diagonal wave K cycles after the
                // previous block. Waves may overlap, but never on one lane.
                for (time_slot = 0;
                     time_slot < ((((tm + 3) >> 2) - 1) * tk + 7);
                     time_slot = time_slot + 1) begin
                    for (lane = 0; lane < cols; lane = lane + 1) begin
                        for (m_block = 0; m_block < ((tm + 3) >> 2); m_block = m_block + 1) begin
                            rows = (tm - m_block * 4 >= 4) ? 4 : tm - m_block * 4;
                            phase = time_slot - m_block * tk;
                            row_in_tile = phase - lane;
                            if ((row_in_tile >= 0) && (row_in_tile < rows)) begin
                                row = m_block * 4 + row_in_tile;
                                col = n_tile * 4 + lane;
                                golden_fifo[golden_fifo_count] =
                                    golden_c_value(row, col, tk, tn, wgt_scale);
                                golden_fifo_row[golden_fifo_count] = row;
                                golden_fifo_col[golden_fifo_count] = col;
                                golden_fifo_count = golden_fifo_count + 1;
                            end
                        end
                    end
                end
            end
        end
    endtask

    task print_case_info;
        input integer case_id;
        input [15:0] tm;
        input [15:0] tk;
        input [15:0] tn;
        input integer wgt_scale;
        begin
            $display("BEGIN: case=%0d A=%0dx%0d B=%0dx%0d wgt_scale=%0d",
                     case_id, tm, tk, tk, tn, wgt_scale);
            if (random_data_mode)
                $display("  deterministic random input seed=%0d, values=0..2", random_seed);
        end
    endtask

    task check_result_capture;
        input integer case_id;
        input [15:0] tm;
        input [15:0] tk;
        input [15:0] tn;
        input integer wgt_scale;
        integer row;
        integer col;
        integer exp_val;
        integer missing_count;
        integer mismatch_count;
        integer util_per_mille;
        integer active_per_mille;
        begin
            util_per_mille = (array_en_cycles == 0) ? 0 :
                ((tm * tn * tk * 1000) / (16 * array_en_cycles));
            active_per_mille = (case_cycles == 0) ? 0 :
                ((array_en_cycles * 1000) / case_cycles);
            $display("UTIL: case=%0d M=%0d K=%0d N=%0d case_cycles=%0d array_en=%0d util=%0d.%03d active=%0d.%03d",
                     case_id, tm, tk, tn, case_cycles, array_en_cycles,
                     util_per_mille / 1000, util_per_mille % 1000,
                     active_per_mille / 1000, active_per_mille % 1000);

            missing_count = 0;
            mismatch_count = 0;
            for (row = 0; row < tm; row = row + 1) begin
                for (col = 0; col < tn; col = col + 1) begin
                    exp_val = golden_c_value(row, col, tk, tn, wgt_scale);
                    if (!result_seen[row][col])
                        missing_count = missing_count + 1;
                    else if (result_mem[row][col] != exp_val)
                        mismatch_count = mismatch_count + 1;
                end
            end

            if ((missing_count != 0) || (mismatch_count != 0) ||
                (result_duplicate_count != 0) ||
                (result_capture_error_count != 0)) begin
                print_case_info(case_id, tm, tk, tn, wgt_scale);
                $display("RESULT CHECK FAIL: case=%0d captured=%0d expected=%0d missing=%0d mismatch=%0d extra=%0d order_error=%0d",
                         case_id, result_capture_count, golden_fifo_count,
                         missing_count, mismatch_count, result_duplicate_count,
                         result_capture_error_count);
                fail_count = fail_count + 1;
            end
            else begin
                pass_count = pass_count + 1;
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
            case_timed_out = 1'b0;
            array_en_cycles = 0;
            case_cycles = 0;
            case_measure_en = 1'b0;
            load_stream_case(tm, tk, tn, act_base, wgt_base);
            clear_result_capture();
            build_golden_fifo(tm, tk, tn, wgt_scale);

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
            case_measure_en = 1'b1;
            @(negedge i_clk);
            i_size_ld = 1'b0;

            wait_done_state();
            case_measure_en = 1'b0;
            if (case_timed_out) begin
                print_case_info(case_id, tm, tk, tn, wgt_scale);
                $display("RESULT CHECK FAIL: case=%0d timeout waiting for S_DONE", case_id);
                fail_count = fail_count + 1;
            end
            else begin
                check_result_capture(case_id, tm, tk, tn, wgt_scale);
                wait_idle_state();
            end
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
            array_en_cycles = 0;
            case_cycles = 0;
            case_measure_en = 1'b0;
            load_stream_case(tm, tk, tn, act_base, wgt_base);
            clear_result_capture();
            build_golden_fifo(tm, tk, tn, wgt_scale);

            if (dut.state != 3'd0) begin
                $display("ERROR: case=%0d expected S_IDLE before no-reset start, state=%0d",
                         case_id, dut.state);
                $finish;
            end

            @(negedge i_clk);
            i_act_base_addr = act_base;
            i_wgt_base_addr = wgt_base;
            i_m_size = tm;
            i_k_size = tk;
            i_n_size = tn;
            i_size_ld = 1'b1;
            case_measure_en = 1'b1;
            @(negedge i_clk);
            i_size_ld = 1'b0;

            wait_done_state();
            case_measure_en = 1'b0;
            check_result_capture(case_id, tm, tk, tn, wgt_scale);
            wait_idle_state();
            repeat (4) @(negedge i_clk);
        end
    endtask

    task run_exhaustive_shape_cases;
        begin
            for (loop_m = 1; loop_m <= 8; loop_m = loop_m + 1) begin
                for (loop_n = 1; loop_n <= 8; loop_n = loop_n + 1) begin
                    for (loop_k = 5; loop_k <= 8; loop_k = loop_k + 1) begin
                        run_case(case_id_next,
                                 loop_m[15:0],
                                 loop_k[15:0],
                                 loop_n[15:0],
                                 32'd100,
                                 32'd200,
                                 1);
                        case_id_next = case_id_next + 1;
                    end
                end
            end
        end
    endtask

    task run_backpressure_cases;
        begin
            wgt_base_delay_cfg = 4'd2;
            run_case(case_id_next, 16'd5, 16'd6, 16'd8, 32'd100, 32'd200, 1);
            case_id_next = case_id_next + 1;

            wgt_base_delay_cfg = 4'd0;
            act_base_delay_cfg = 4'd2;
            run_case(case_id_next, 16'd5, 16'd6, 16'd8, 32'd100, 32'd200, 1);
            case_id_next = case_id_next + 1;

            act_base_delay_cfg = 4'd1;
            wgt_base_delay_cfg = 4'd3;
            run_case(case_id_next, 16'd8, 16'd7, 16'd8, 32'd100, 32'd200, 1);
            case_id_next = case_id_next + 1;

            act_base_delay_cfg = 4'd3;
            wgt_base_delay_cfg = 4'd1;
            run_case(case_id_next, 16'd3, 16'd5, 16'd6, 32'd100, 32'd200, 1);
            case_id_next = case_id_next + 1;

            act_base_delay_cfg = 4'd0;
            wgt_base_delay_cfg = 4'd0;
        end
    endtask

    task run_large_shape_cases;
        begin
            run_case(case_id_next, 16'd4,  16'd5,  16'd16, 32'd100, 32'd200, 1);
            case_id_next = case_id_next + 1;
            run_case(case_id_next, 16'd8,  16'd8,  16'd12, 32'd100, 32'd200, 1);
            case_id_next = case_id_next + 1;
            run_case(case_id_next, 16'd7,  16'd9,  16'd7,  32'd100, 32'd200, 1);
            case_id_next = case_id_next + 1;
            run_case(case_id_next, 16'd7,  16'd12, 16'd7,  32'd100, 32'd200, 1);
            case_id_next = case_id_next + 1;
            run_case(case_id_next, 16'd16, 16'd5,  16'd16, 32'd100, 32'd200, 1);
            case_id_next = case_id_next + 1;
            run_case(case_id_next, 16'd12, 16'd8,  16'd12, 32'd100, 32'd200, 1);
            case_id_next = case_id_next + 1;
            run_case(case_id_next, 16'd9,  16'd9,  16'd9,  32'd100, 32'd200, 1);
            case_id_next = case_id_next + 1;
            run_case(case_id_next, 16'd9,  16'd12, 16'd9,  32'd100, 32'd200, 1);
            case_id_next = case_id_next + 1;
            run_case(case_id_next, 16'd7,  16'd16, 16'd7,  32'd100, 32'd200, 1);
            case_id_next = case_id_next + 1;
        end
    endtask

    task run_random_large_shape_cases;
        begin
            random_data_mode = 1'b1;
            for (loop_m = 100; loop_m <= 103; loop_m = loop_m + 1) begin
                for (loop_k = 36; loop_k <= 39; loop_k = loop_k + 1) begin
                    for (loop_n = 7; loop_n <= 10; loop_n = loop_n + 1) begin
                        random_seed = case_id_next;
                        run_case(case_id_next,
                                 loop_m[15:0],
                                 loop_k[15:0],
                                 loop_n[15:0],
                                 32'd100,
                                 32'd200,
                                 1);
                        case_id_next = case_id_next + 1;
                    end
                end
            end
            random_data_mode = 1'b0;
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
        end
        else if (o_act_base_valid) begin
            act_pending_base_addr <= o_act_base_addr;
            act_base_delay_left <= (act_base_delay_cfg == 4'b0) ? 4'd1 : act_base_delay_cfg;
            act_base_pending <= 1'b1;
            i_act_stream_valid <= 1'b0;
            i_act_stream_data <= {4*DATA_W{1'b0}};
            act_loader_busy <= 1'b0;
        end
        else if (act_base_pending) begin
            if (act_base_delay_left == 4'd1) begin
                act_rd_addr <= act_pending_base_addr;
                act_words_left <= i_k_size * ((i_m_size + 16'd3) >> 2);
                i_act_stream_data <= act_sram[act_pending_base_addr];
                i_act_stream_valid <= 1'b1;
                act_loader_busy <= 1'b1;
                act_base_pending <= 1'b0;
                act_base_delay_left <= 4'b0;
            end
            else begin
                act_base_delay_left <= act_base_delay_left - 1'b1;
            end
        end
        else if (i_act_stream_valid && o_act_stream_ready) begin
            if (act_words_left == 16'd1) begin
                i_act_stream_valid <= 1'b0;
                i_act_stream_data <= {4*DATA_W{1'b0}};
                act_words_left <= 16'b0;
                act_loader_busy <= 1'b0;
            end
            else begin
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
        end
        else if (o_wgt_base_valid) begin
            wgt_pending_base_addr <= o_wgt_base_addr;
            wgt_base_delay_left <= (wgt_base_delay_cfg == 4'b0) ? 4'd1 : wgt_base_delay_cfg;
            wgt_base_pending <= 1'b1;
            i_wgt_stream_valid <= 1'b0;
            i_wgt_stream_data <= {4*DATA_W{1'b0}};
            wgt_loader_busy <= 1'b0;
        end
        else if (wgt_base_pending) begin
            if (wgt_base_delay_left == 4'd1) begin
                wgt_rd_addr <= wgt_pending_base_addr;
                wgt_words_left <= i_k_size;
                i_wgt_stream_data <= wgt_sram[wgt_pending_base_addr];
                i_wgt_stream_valid <= 1'b1;
                wgt_loader_busy <= 1'b1;
                wgt_base_pending <= 1'b0;
                wgt_base_delay_left <= 4'b0;
            end
            else begin
                wgt_base_delay_left <= wgt_base_delay_left - 1'b1;
            end
        end
        else if (i_wgt_stream_valid && o_wgt_stream_ready) begin
            if (wgt_words_left == 16'd1) begin
                i_wgt_stream_valid <= 1'b0;
                i_wgt_stream_data <= {4*DATA_W{1'b0}};
                wgt_words_left <= 16'b0;
                wgt_loader_busy <= 1'b0;
            end
            else begin
                wgt_rd_addr <= wgt_rd_addr + 1'b1;
                wgt_words_left <= wgt_words_left - 1'b1;
                i_wgt_stream_data <= wgt_sram[wgt_rd_addr + 1'b1];
            end
        end
    end

    always @(posedge i_clk) begin
        if (i_rst_n && case_measure_en)
            case_cycles = case_cycles + 1;
        if (i_rst_n && case_measure_en && o_array_en)
            array_en_cycles = array_en_cycles + 1;

        if (i_rst_n && (dut.state == 3'd2) &&
            i_act_stream_valid_dut && o_act_stream_ready &&
            !(i_wgt_stream_valid_dut && o_wgt_stream_ready) && !dut.wgt_all_read_done) begin
            $display("ERROR: case=%0d cycle=%0d act advanced while waiting for wgt stream",
                     current_case, cycle);
            $finish;
        end
    end

    always @(posedge i_clk) begin
        if (i_rst_n && (dut.state == 3'd2) &&
            i_wgt_stream_valid_dut && o_wgt_stream_ready &&
            !(i_act_stream_valid_dut && o_act_stream_ready) && !dut.wgt_all_read_done) begin
            $display("ERROR: case=%0d cycle=%0d wgt advanced while waiting for act stream",
                     current_case, cycle);
            $finish;
        end
    end

    // Test-only result scoreboard. Psum values arrive tile by tile in
    // diagonal order: N tile, M block, phase, then lane.
    always @(negedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            for (result_i = 0; result_i < RESULT_SLOTS; result_i = result_i + 1) begin
                result_active[result_i] = 1'b0;
                result_slot_mask[result_i] = 4'b0;
            end
        end
        else if (current_case != 0) begin
            for (result_j = 0; result_j < 4; result_j = result_j + 1) begin
                integer val;
                integer row;
                integer col;
                integer exp_val;

                if (o_psum_valid[result_j]) begin
                    val = psum_lane_value(result_j);
                    if (result_capture_count >= golden_fifo_count) begin
                        result_duplicate_count = result_duplicate_count + 1;
                        $display("EXTRA PSUM SAMPLE: case=%0d cycle=%0d lane=%0d value=%0d after expected output",
                                 current_case, cycle, result_j, val);
                    end
                    else begin
                        row = golden_fifo_row[result_capture_count];
                        col = golden_fifo_col[result_capture_count];
                        exp_val = golden_fifo[result_capture_count];
                        result_mem[row][col] = val;
                        result_seen[row][col] = 1'b1;
                        if (val != exp_val) begin
                            result_capture_error_count = result_capture_error_count + 1;
                            $display("PSUM ORDER ERROR: case=%0d cycle=%0d lane=%0d C[%0d][%0d] expected=%0d observed=%0d valid=%b sel=%h zero=%h",
                                     current_case, cycle, result_j, row, col,
                                     exp_val, val, o_psum_valid, o_psum_sel, o_psum_zero);
                        end
                    end
                    result_capture_count = result_capture_count + 1;
                end
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
        pass_count = 0;
        fail_count = 0;
        case_timed_out = 1'b0;
        array_en_cycles = 0;
        case_cycles = 0;
        case_measure_en = 1'b0;
        case_id_next = 1;
        wgt_value_scale = 1;
        act_base_delay_cfg = 4'd0;
        wgt_base_delay_cfg = 4'd0;
        random_data_mode = 1'b0;
        random_seed = 0;

        run_exhaustive_shape_cases();
        run_backpressure_cases();
        run_large_shape_cases();
        run_case_no_reset(case_id_next, 16'd2, 16'd5, 16'd2, 32'd100, 32'd200, 1);
        case_id_next = case_id_next + 1;
        run_random_large_shape_cases();

        $display("TEST SUMMARY: total_cases=%0d pass=%0d fail=%0d",
                 pass_count + fail_count, pass_count, fail_count);
        $finish;
    end
endmodule
