`timescale 1ns / 1ps

module trans_wgt #(
    parameter DATA_W = 8
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     hold,

    input  wire                     valid,

    input  wire [15:0]              k_size,
    input  wire [15:0]              n_size,

    input  wire [4*DATA_W-1:0]      in_data,
    input  wire                     in_data_valid,
    output wire                     in_data_ready,

    output reg  [4*DATA_W-1:0]      out_data,
    output reg                      out_data_valid,
    input  wire                     out_data_ready,

    output reg                      block_done,
    output reg                      done
);

    localparam S_WAIT_BLOCK = 2'd0;
    localparam S_RUN        = 2'd1;
    localparam S_DONE       = 2'd2;

    reg [1:0] state;

    reg signed [DATA_W-1:0] row_buf [0:3][0:3];
    reg [15:0]              row_tag [0:3];
    reg                     row_valid [0:3];

    reg [15:0] n_base;
    reg [15:0] block_cols;
    reg [15:0] load_k;
    reg [15:0] out_idx;
    reg [15:0] last_out_idx;

    reg [15:0] row_idx;
    reg [1:0]  row_slot;
    reg        can_load;
    reg        can_output;
    reg        last_block;
    reg [15:0] load_wait_idx;
    reg        replay_pending;
    reg [15:0] load_idx;
    reg        load_wrap_now;
    wire       out_slot_ready;

    integer lane;
    integer r;
    integer c;

    always @(*) begin
        if ((n_size - n_base) >= 4)
            block_cols = 4;
        else
            block_cols = n_size - n_base;

        if (block_cols == 0)
            last_out_idx = 0;
        else
            last_out_idx = k_size + block_cols - 2;

        last_block = ((n_base + 4) >= n_size);

        load_wrap_now = replay_pending && (load_k >= k_size);
        load_idx = load_wrap_now ? 16'd0 : load_k;

        can_output = !hold &&
                     out_slot_ready &&
                     !load_wrap_now &&
                     (block_cols != 0) &&
                     (out_idx <= last_out_idx) &&
                     ((out_idx != 0) || (load_k >= block_cols)) &&
                     ((out_idx >= k_size) || (load_k > out_idx));

        load_wait_idx = 0;
        if ((load_idx + block_cols) > 5)
            load_wait_idx = load_idx + block_cols - 5;

        if (replay_pending && (load_idx < block_cols)) begin
            can_load = !hold &&
                       out_slot_ready &&
                       (state == S_RUN) &&
                       ((out_idx >= (k_size + load_idx)) ||
                        (out_idx < block_cols));
        end else begin
            can_load = !hold &&
                       out_slot_ready &&
                       (state == S_RUN) &&
                   ((load_k < k_size) || load_wrap_now) &&
                       ((load_idx < block_cols) ||
                        ((load_idx + block_cols) <= 5) ||
                        (out_idx >= load_wait_idx));
        end
    end

    assign out_slot_ready = !out_data_valid || out_data_ready;
    assign in_data_ready = can_load;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_WAIT_BLOCK;
            out_data <= {4*DATA_W{1'b0}};
            out_data_valid <= 1'b0;
            block_done <= 1'b0;
            done <= 1'b0;
            n_base <= 0;
            load_k <= 0;
            out_idx <= 0;
            replay_pending <= 1'b0;

            for (r = 0; r < 4; r = r + 1) begin
                row_tag[r] <= 0;
                row_valid[r] <= 1'b0;
                for (c = 0; c < 4; c = c + 1)
                    row_buf[r][c] <= {DATA_W{1'b0}};
            end
        end else if (hold) begin
        end else begin
            case (state)
                S_WAIT_BLOCK: begin
                    out_data <= {4*DATA_W{1'b0}};
                    out_data_valid <= 1'b0;
                    done <= 1'b0;

                    if (valid) begin
                        block_done <= 1'b0;
                        load_k <= 0;
                        out_idx <= 0;
                        replay_pending <= 1'b0;
                        for (r = 0; r < 4; r = r + 1) begin
                            row_tag[r] <= 0;
                            row_valid[r] <= 1'b0;
                        end
                        state <= S_RUN;
                    end
                end

                S_RUN: begin
                    block_done <= 1'b0;

                    if (valid)
                        replay_pending <= 1'b1;

                    if (!valid && in_data_valid && can_load) begin
                        row_slot = load_idx[1:0];
                        row_tag[row_slot] <= load_idx;
                        row_valid[row_slot] <= 1'b1;

                        for (lane = 0; lane < 4; lane = lane + 1) begin
                            if (lane < block_cols)
                                row_buf[row_slot][lane] <=
                                    in_data[lane*DATA_W +: DATA_W];
                            else
                                row_buf[row_slot][lane] <= {DATA_W{1'b0}};
                        end

                        if (load_wrap_now)
                            replay_pending <= 1'b1;
                        else if (replay_pending && (load_idx == (block_cols - 1)))
                            replay_pending <= 1'b0;
                        if (load_wrap_now)
                            out_idx <= 0;
                        load_k <= load_idx + 1;
                    end

                    if (!valid && can_output) begin
                        out_data <= {4*DATA_W{1'b0}};
                        out_data_valid <= 1'b1;

                        for (lane = 0; lane < 4; lane = lane + 1) begin
                            if ((lane < block_cols) && (out_idx >= lane)) begin
                                row_idx = out_idx - lane;
                                row_slot = row_idx[1:0];
                                if ((row_idx < k_size) &&
                                    row_valid[row_slot] &&
                                    (row_tag[row_slot] == row_idx))
                                    out_data[lane*DATA_W +: DATA_W] <=
                                        row_buf[row_slot][lane];
                            end
                        end

                        if (out_idx == last_out_idx) begin
                            if (replay_pending || (load_k < k_size)) begin
                                out_idx <= 0;
                            end else if (last_block) begin
                                done <= 1'b1;
                                state <= S_DONE;
                            end else begin
                                n_base <= n_base + 4;
                                block_done <= 1'b1;
                                state <= S_WAIT_BLOCK;
                            end
                        end else begin
                            out_idx <= out_idx + 1;
                        end
                    end else if (out_data_valid && out_data_ready) begin
                        out_data_valid <= 1'b0;
                    end
                end

                S_DONE: begin
                    out_data_valid <= 1'b0;
                    block_done <= 1'b0;
                    done <= 1'b1;

                    if (valid) begin
                        done <= 1'b0;
                        load_k <= 0;
                        out_idx <= 0;
                        replay_pending <= 1'b0;
                        for (r = 0; r < 4; r = r + 1) begin
                            row_tag[r] <= 0;
                            row_valid[r] <= 1'b0;
                        end
                        state <= S_RUN;
                    end
                end

                default: begin
                    state <= S_WAIT_BLOCK;
                end
            endcase
        end
    end

endmodule
