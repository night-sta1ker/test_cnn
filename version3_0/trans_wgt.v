`timescale 1ns / 1ps

module trans_wgt #(
    parameter DATA_W = 8
) (
    input  wire                     clk,
    input  wire                     rst_n,

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
    reg        replay_pending;
    reg        replay_load_started;
    reg        wait_replay_rows;
    reg [15:0] replay_base;
    reg [15:0] load_idx;
    reg        restart_active;
    reg [1:0]  load_slot;
    reg [15:0] slot_last_use;
    reg        slot_safe_to_write;
    reg [4*DATA_W-1:0] out_data_next;
    reg [15:0] out_row_idx;
    reg [1:0]  out_row_slot;
    reg        out_data_has_valid;

    integer load_lane;
    integer out_lane;
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

        restart_active = replay_pending || ((state == S_RUN) && valid);
        load_idx = load_k;

        load_slot = load_idx[1:0];
        slot_last_use = row_tag[load_slot] + block_cols - 1'b1;
        slot_safe_to_write = !row_valid[load_slot] ||
                             wait_replay_rows ||
                             (out_idx >= slot_last_use);

        can_load = (state == S_RUN) &&
                   ((load_k < k_size) || replay_pending || ((state == S_RUN) && valid)) &&
                   slot_safe_to_write;

        out_data_next = {4*DATA_W{1'b0}};
        out_data_has_valid = 1'b0;
        for (out_lane = 0; out_lane < 4; out_lane = out_lane + 1) begin
            if ((out_lane < block_cols) && (out_idx >= out_lane)) begin
                out_row_idx = out_idx - out_lane;
                out_row_slot = out_row_idx[1:0];
                if ((out_row_idx < k_size) &&
                    row_valid[out_row_slot] &&
                    (row_tag[out_row_slot] == out_row_idx)) begin
                    out_data_next[out_lane*DATA_W +: DATA_W] =
                        row_buf[out_row_slot][out_lane];
                    out_data_has_valid = 1'b1;
                end
            end
        end

        can_output = out_data_ready &&
                     !wait_replay_rows &&
                     out_data_has_valid &&
                     (block_cols != 0) &&
                     ((out_idx != 0) || (load_k >= block_cols)) &&
                     ((out_idx >= k_size) || (load_k > out_idx));
    end

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
            replay_load_started <= 1'b0;
            wait_replay_rows <= 1'b0;
            replay_base <= 0;

            for (r = 0; r < 4; r = r + 1) begin
                row_tag[r] <= 0;
                row_valid[r] <= 1'b0;
                for (c = 0; c < 4; c = c + 1)
                    row_buf[r][c] <= {DATA_W{1'b0}};
            end
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
                        replay_load_started <= 1'b0;
                        wait_replay_rows <= 1'b0;
                        replay_base <= 0;
                        for (r = 0; r < 4; r = r + 1) begin
                            row_tag[r] <= 0;
                            row_valid[r] <= 1'b0;
                        end
                        state <= S_RUN;
                    end
                end

                S_RUN: begin
                    block_done <= 1'b0;

                    if (valid) begin
                        replay_pending <= 1'b1;
                        replay_load_started <= 1'b0;
                        wait_replay_rows <= 1'b0;
                        replay_base <= load_k;
                    end

                    if (in_data_valid && can_load) begin
                        row_tag[load_slot] <= load_idx;
                        row_valid[load_slot] <= 1'b1;

                        for (load_lane = 0; load_lane < 4; load_lane = load_lane + 1) begin
                            if (load_lane < block_cols)
                                row_buf[load_slot][load_lane] <=
                                    in_data[load_lane*DATA_W +: DATA_W];
                            else
                                row_buf[load_slot][load_lane] <= {DATA_W{1'b0}};
                        end

                        if (restart_active)
                            replay_load_started <= 1'b1;
                        if (replay_pending && (load_idx == (replay_base + block_cols - 1))) begin
                            replay_pending <= 1'b0;
                            replay_load_started <= 1'b0;
                        end
                        load_k <= load_idx + 1;
                    end

                    if (can_output) begin
                        out_data <= out_data_next;
                        out_data_valid <= 1'b1;

                        if (out_idx == last_out_idx) begin
                            if (replay_pending || (load_k < k_size)) begin
                                out_idx <= out_idx + 1;
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
                    end else begin
                        out_data_valid <= 1'b0;
                    end

                    if (wait_replay_rows && (load_k >= block_cols)) begin
                        wait_replay_rows <= 1'b0;
                        out_idx <= 0;
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
                        replay_load_started <= 1'b0;
                        wait_replay_rows <= 1'b0;
                        replay_base <= 0;
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
