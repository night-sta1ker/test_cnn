`timescale 1ns / 1ps

// ============================================================================
// Sequencer_1
//
// FSM version based on row trackers:
//   - Iterate over B/C columns in 4-column blocks.
//   - Stream A rows for the current column block.
//   - Each launched A row creates one tracker. The tracker records where the
//     row entered the array and when its 4 outputs become valid.
// ============================================================================

module Sequencer_1 #(
    parameter DATA_W = 8,
    parameter ACC_W  = 32,
    parameter MAX_M_SIZE = 4,
    parameter MAX_K_SIZE = 4,
    parameter MAX_N_SIZE = 4
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire [31:0]              m_size,
    input  wire [31:0]              k_size,
    input  wire [31:0]              n_size,

    output reg  [3:0]               wgt_ld,
    output reg  [4*DATA_W-1:0]      wgt_in,
    output reg                      en,
    output reg                      clear,
    output reg  [4*DATA_W-1:0]      act_in,
    output reg  [15:0]              psum_zero,
    output reg  [1:0]               psum_out_sel,

    input  wire [4*ACC_W-1:0]       psum_out,

    output reg                      done,
    output reg  [31:0]              cycle,
    output wire [MAX_M_SIZE*MAX_N_SIZE*ACC_W-1:0] c_out_flat
);

    localparam S_IDLE      = 2'd0;
    localparam S_RUN_BLOCK = 2'd1;
    localparam S_WAIT_ROWS = 2'd2;
    localparam S_DONE      = 2'd3;

    reg signed [DATA_W-1:0] a_mem [0:MAX_M_SIZE-1][0:MAX_K_SIZE-1];
    reg signed [DATA_W-1:0] b_mem [0:MAX_K_SIZE-1][0:MAX_N_SIZE-1];
    reg signed [ACC_W-1:0]  c_mem [0:MAX_M_SIZE-1][0:MAX_N_SIZE-1];

    reg [1:0]  state;

    reg [31:0] col_base;
    reg [2:0]  block_cols;

    reg [31:0] batch_age;
    reg [1:0]  batch_start_row;
    reg [31:0] next_a_row;
    reg [31:0] launch_slot;

    reg                     row_valid [0:MAX_M_SIZE-1];
    reg [31:0]              row_age   [0:MAX_M_SIZE-1];
    reg [31:0]              row_m     [0:MAX_M_SIZE-1];
    reg [31:0]              row_n0    [0:MAX_M_SIZE-1];
    reg [2:0]               row_cols  [0:MAX_M_SIZE-1];
    reg [1:0]               row_start [0:MAX_M_SIZE-1];

    integer i;
    integer j;
    integer k;
    integer r;
    integer c;

    reg [31:0] k_idx;
    reg [31:0] global_n;
    reg [1:0]  phase_idx;
    reg [1:0]  act_row_idx;
    reg [1:0]  final_row;
    reg        launch_row_fire;
    reg        any_row_active;

    genvar gr;
    genvar gc;
    generate
        for (gr = 0; gr < MAX_M_SIZE; gr = gr + 1) begin : gen_c_flat_row
            for (gc = 0; gc < MAX_N_SIZE; gc = gc + 1) begin : gen_c_flat_col
                assign c_out_flat[(gr*MAX_N_SIZE + gc)*ACC_W +: ACC_W] = c_mem[gr][gc];
            end
        end
    endgenerate

    initial begin
        for (i = 0; i < MAX_M_SIZE; i = i + 1) begin
            for (k = 0; k < MAX_K_SIZE; k = k + 1)
                a_mem[i][k] = ((i*3 + k*5 + 1) % 9) - 4;
        end

        for (k = 0; k < MAX_K_SIZE; k = k + 1) begin
            for (j = 0; j < MAX_N_SIZE; j = j + 1)
                b_mem[k][j] = ((k*2 + j*3 + 2) % 7) - 3;
        end

        for (i = 0; i < MAX_M_SIZE; i = i + 1) begin
            for (j = 0; j < MAX_N_SIZE; j = j + 1)
                c_mem[i][j] = {ACC_W{1'b0}};
        end
    end

    always @(*) begin
        en              = rst_n && (state != S_IDLE) && (state != S_DONE);
        clear           = 1'b0;
        wgt_ld          = 4'b0000;
        wgt_in          = {4*DATA_W{1'b0}};
        act_in          = {4*DATA_W{1'b0}};
        psum_zero       = 16'h0000;
        psum_out_sel    = 2'd0;
        launch_row_fire = 1'b0;

        if (rst_n && (state == S_RUN_BLOCK)) begin
            for (c = 0; c < 4; c = c + 1) begin
                k_idx = batch_age - c;
                if ((c < block_cols) &&
                    (batch_age >= c) &&
                    (k_idx < k_size)) begin
                    global_n = col_base + c;
                    phase_idx = batch_start_row + k_idx[1:0] + c[1:0];
                    wgt_in[c*DATA_W +: DATA_W] = b_mem[k_idx][global_n];
                    wgt_ld[phase_idx] = 1'b1;
                end
            end

            if ((batch_age >= 1) && (batch_age <= 4) && (next_a_row < m_size)) begin
                launch_row_fire = 1'b1;
                k_idx = 0;
                act_row_idx = batch_start_row;
                act_in[act_row_idx*DATA_W +: DATA_W] = a_mem[next_a_row][0];

                for (c = 0; c < 4; c = c + 1) begin
                    if (c < block_cols)
                        psum_zero[batch_start_row*4 + c] = (c == 0);
                end
            end
        end

        if (rst_n && (state != S_IDLE) && (state != S_DONE)) begin
            for (r = 0; r < MAX_M_SIZE; r = r + 1) begin
                if (row_valid[r]) begin
                    k_idx = row_age[r];
                    if ((k_idx < k_size) && (k_idx != 0)) begin
                        act_row_idx = row_start[r] + k_idx[1:0];
                        act_in[act_row_idx*DATA_W +: DATA_W] = a_mem[row_m[r]][k_idx];
                    end

                    for (c = 0; c < 4; c = c + 1) begin
                        if ((c < row_cols[r]) && (row_age[r] == c))
                            psum_zero[row_start[r]*4 + c] = 1'b1;

                        if ((c < row_cols[r]) &&
                            ((row_age[r] == (k_size - 1 + c)) ||
                             (row_age[r] == (k_size + c)))) begin
                            final_row = row_start[r] + k_size[1:0] - 1'b1;
                            psum_out_sel = final_row;
                        end
                    end
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            col_base        <= 0;
            block_cols      <= 0;
            batch_age       <= 0;
            batch_start_row <= 0;
            next_a_row      <= 0;
            launch_slot     <= 0;
            cycle           <= 0;
            done            <= 1'b0;

            for (r = 0; r < MAX_M_SIZE; r = r + 1) begin
                row_valid[r] <= 1'b0;
                row_age[r]   <= 0;
                row_m[r]     <= 0;
                row_n0[r]    <= 0;
                row_cols[r]  <= 0;
                row_start[r] <= 0;
            end

            for (i = 0; i < MAX_M_SIZE; i = i + 1) begin
                for (j = 0; j < MAX_N_SIZE; j = j + 1)
                    c_mem[i][j] <= {ACC_W{1'b0}};
            end
        end else begin
            case (state)
                S_IDLE: begin
                    col_base        <= 0;
                    block_cols      <= (n_size >= 4) ? 4 : n_size[2:0];
                    batch_age       <= 0;
                    batch_start_row <= 0;
                    next_a_row      <= 0;
                    launch_slot     <= 0;
                    cycle           <= 0;
                    done            <= 1'b0;
                    state           <= S_RUN_BLOCK;
                end

                S_RUN_BLOCK: begin
                    any_row_active = 1'b0;

                    for (r = 0; r < MAX_M_SIZE; r = r + 1) begin
                        if (row_valid[r]) begin
                            for (c = 0; c < 4; c = c + 1) begin
                                if ((c < row_cols[r]) && (row_age[r] == (k_size + c))) begin
                                    c_mem[row_m[r]][row_n0[r] + c] <=
                                        $signed(psum_out[c*ACC_W +: ACC_W]);
                                end
                            end

                            if (row_age[r] >= (k_size + row_cols[r] - 1)) begin
                                row_valid[r] <= 1'b0;
                            end else begin
                                row_age[r] <= row_age[r] + 1;
                                any_row_active = 1'b1;
                            end
                        end
                    end

                    if (launch_row_fire) begin
                        row_valid[launch_slot] <= 1'b1;
                        row_age[launch_slot]   <= 1;
                        row_m[launch_slot]     <= next_a_row;
                        row_n0[launch_slot]    <= col_base;
                        row_cols[launch_slot]  <= block_cols;
                        row_start[launch_slot] <= batch_start_row;

                        next_a_row  <= next_a_row + 1;
                        launch_slot <= (launch_slot == (MAX_M_SIZE - 1)) ? 0 : (launch_slot + 1);
                        any_row_active = 1'b1;
                    end

                    if (batch_age == (k_size - 1)) begin
                        batch_age       <= 0;
                        batch_start_row <= batch_start_row + k_size[1:0];
                    end else begin
                        batch_age <= batch_age + 1;
                    end

                    if (next_a_row >= m_size) begin
                        state <= S_WAIT_ROWS;
                    end

                    cycle <= cycle + 1;
                end

                S_WAIT_ROWS: begin
                    any_row_active = 1'b0;

                    for (r = 0; r < MAX_M_SIZE; r = r + 1) begin
                        if (row_valid[r]) begin
                            for (c = 0; c < 4; c = c + 1) begin
                                if ((c < row_cols[r]) && (row_age[r] == (k_size + c))) begin
                                    c_mem[row_m[r]][row_n0[r] + c] <=
                                        $signed(psum_out[c*ACC_W +: ACC_W]);
                                end
                            end

                            if (row_age[r] >= (k_size + row_cols[r] - 1)) begin
                                row_valid[r] <= 1'b0;
                            end else begin
                                row_age[r] <= row_age[r] + 1;
                                any_row_active = 1'b1;
                            end
                        end
                    end

                    if (!any_row_active) begin
                        if ((col_base + 4) < n_size) begin
                            col_base        <= col_base + 4;
                            block_cols      <= ((n_size - (col_base + 4)) >= 4) ? 4 : (n_size - (col_base + 4));
                            batch_age       <= 0;
                            batch_start_row <= 0;
                            next_a_row      <= 0;
                            launch_slot     <= 0;
                            state           <= S_RUN_BLOCK;
                        end else begin
                            done  <= 1'b1;
                            state <= S_DONE;
                        end
                    end

                    cycle <= cycle + 1;
                end

                S_DONE: begin
                    done <= 1'b1;
                end
            endcase
        end
    end

endmodule
