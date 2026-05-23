`timescale 1ns / 1ps

module trans_act #(
    parameter DATA_W = 8,
    parameter MAX_M_SIZE = 128,
    parameter MAX_K_SIZE = 128
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     hold,
    input  wire                     clear,

    input  wire [15:0]              m_size,
    input  wire [15:0]              k_size,

    input  wire [4*DATA_W-1:0]      in_data,
    input  wire                     in_data_valid,
    output reg                      in_data_ready,

    output reg  [4*DATA_W-1:0]      out_data,
    output reg                      out_data_valid,
    input  wire                     out_data_ready,

    output reg                      done
);

    localparam S_IDLE = 2'd0;
    localparam S_RUN  = 2'd1;
    localparam S_DONE = 2'd2;
    localparam INVALID_CYCLE = 16'hffff;

    reg [1:0] state;

    reg signed [DATA_W-1:0] row_buf [0:7][0:MAX_K_SIZE-1];
    reg [15:0]              row_start_cycle [0:7];

    reg [15:0] load_m;
    reg [15:0] load_k;
    reg [15:0] out_cycle;

    reg [15:0] load_m_next;
    reg [15:0] load_k_next;

    reg [2:0]  load_slot;
    reg [2:0]  out_slot;
    reg [15:0] out_k;
    reg [1:0]  out_lane;
    reg        row_active;
    reg        missing_required_row;
    reg        any_row_left;
    reg        output_this_cycle;
    wire       out_slot_ready;

    integer lane;
    integer slot;
    integer check_m;
    reg [2:0]  check_slot;
    reg [15:0] check_start_cycle;

    assign out_slot_ready = !out_data_valid || out_data_ready;

    always @(*) begin
        missing_required_row = 1'b0;

        for (check_m = 0; check_m < MAX_M_SIZE; check_m = check_m + 1) begin
            if (check_m < m_size) begin
                check_slot = check_m[2:0];
                check_start_cycle = (check_m / 4) * k_size + check_m[1:0];

                if ((out_cycle >= check_start_cycle) &&
                    (out_cycle < (check_start_cycle + k_size)) &&
                    (row_start_cycle[check_slot] != check_start_cycle))
                    missing_required_row = 1'b1;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            in_data_ready <= 1'b0;
            out_data <= {4*DATA_W{1'b0}};
            out_data_valid <= 1'b0;
            done <= 1'b0;
            load_m <= 0;
            load_k <= 0;
            out_cycle <= 0;

            for (slot = 0; slot < 8; slot = slot + 1) begin
                row_start_cycle[slot] <= INVALID_CYCLE;
            end
        end else if (hold) begin
            in_data_ready <= 1'b0;
        end else if (clear) begin
            state <= S_RUN;
            in_data_ready <= 1'b0;
            out_data <= {4*DATA_W{1'b0}};
            out_data_valid <= 1'b0;
            done <= 1'b0;
            load_m <= 0;
            load_k <= 0;
            out_cycle <= 0;

            for (slot = 0; slot < 8; slot = slot + 1) begin
                row_start_cycle[slot] <= INVALID_CYCLE;
            end
        end else begin
            case (state)
                S_IDLE: begin
                    in_data_ready <= 1'b0;
                    out_data_valid <= 1'b0;
                    done <= 1'b0;
                end

                S_RUN: begin
                    in_data_ready <= (load_m < m_size);

                    if (in_data_valid && (load_m < m_size)) begin
                        load_m_next = load_m;
                        load_k_next = load_k;

                        for (lane = 0; lane < 4; lane = lane + 1) begin
                            if (load_m_next < m_size) begin
                                load_slot = load_m_next[2:0];
                                row_buf[load_slot][load_k_next] <=
                                    in_data[lane*DATA_W +: DATA_W];

                                if (load_k_next == (k_size - 1)) begin
                                    row_start_cycle[load_slot] <=
                                        load_m_next[15:2] * k_size +
                                        load_m_next[1:0];

                                    load_k_next = 0;
                                    load_m_next = load_m_next + 1;
                                end else begin
                                    load_k_next = load_k_next + 1;
                                end
                            end
                        end

                        load_m <= load_m_next;
                        load_k <= load_k_next;
                    end

                    if (out_slot_ready && !missing_required_row) begin
                        out_data <= {4*DATA_W{1'b0}};
                        out_data_valid <= 1'b0;
                        output_this_cycle = 1'b0;
                        any_row_left = 1'b0;

                        for (slot = 0; slot < 8; slot = slot + 1) begin
                            out_slot = slot[2:0];
                            row_active = (row_start_cycle[out_slot] != INVALID_CYCLE) &&
                                         (out_cycle >= row_start_cycle[out_slot]) &&
                                         (out_cycle < (row_start_cycle[out_slot] + k_size));

                            if (row_active) begin
                                out_k = out_cycle - row_start_cycle[out_slot];
                                out_lane = row_start_cycle[out_slot][1:0] -
                                           out_slot[1:0] + out_k[1:0];
                                out_data[out_lane*DATA_W +: DATA_W] <=
                                    row_buf[out_slot][out_k];
                                out_data_valid <= 1'b1;
                                output_this_cycle = 1'b1;

                                if (out_k == (k_size - 1))
                                    row_start_cycle[out_slot] <= INVALID_CYCLE;
                                else
                                    any_row_left = 1'b1;
                            end else if (row_start_cycle[out_slot] != INVALID_CYCLE) begin
                                any_row_left = 1'b1;
                            end
                        end

                        out_cycle <= out_cycle + 1;

                        if ((load_m >= m_size) &&
                            !any_row_left &&
                            !output_this_cycle) begin
                            in_data_ready <= 1'b0;
                            done <= 1'b1;
                            state <= S_DONE;
                        end
                    end else if (out_data_valid && out_data_ready) begin
                        out_data_valid <= 1'b0;
                    end
                end

                S_DONE: begin
                    in_data_ready <= 1'b0;
                    out_data_valid <= 1'b0;
                    done <= 1'b1;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
