`timescale 1ns / 1ps

module trans_act #(
    parameter DATA_W = 8,
    parameter MAX_M_SIZE = 128,
    parameter MAX_K_SIZE = 128
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     restart,

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

    reg [1:0] state;

    reg                     row_ready [0:7];
    reg                     row_stream [0:7];
    reg [15:0]              row_m [0:7];
    reg [15:0]              row_k_out [0:7];
    reg [1:0]               row_start_lane [0:7];

    reg [15:0] load_m;
    reg [15:0] load_k;
    reg [15:0] launch_m;
    reg [15:0] launch_group_cycle;
    reg [15:0] out_cycle;

    integer i;
    integer k;
    integer lane;
    integer slot;

    reg [15:0] load_m_tmp;
    reg [15:0] load_k_tmp;
    reg [2:0]  load_slot;
    reg [2:0]  launch_slot;
    reg [2:0]  stream_slot;
    reg [1:0]  start_lane;
    reg [1:0]  out_lane;
    reg [15:0] launch_cycle;
    reg        can_step_out;
    reg        launch_now;
    reg        any_stream_next;

    reg [3:0]               ram_wr_en [0:7];
    reg [4*16-1:0]          ram_wr_k [0:7];
    reg [4*DATA_W-1:0]      ram_wr_data [0:7];
    reg                     ram_rd_en [0:7];
    reg [15:0]              ram_rd_k [0:7];
    reg [1:0]               ram_rd_lane [0:7];
    wire signed [DATA_W-1:0] ram_rd_data [0:7];
    reg                     rd_valid_q [0:7];
    reg [1:0]               rd_lane_q [0:7];

    genvar ram_i;
    generate
        for (ram_i = 0; ram_i < 8; ram_i = ram_i + 1) begin : gen_act_line_ram
            act_line_ram #(
                .DATA_W(DATA_W),
                .MAX_K_SIZE(MAX_K_SIZE)
            ) line_u (
                .clk(clk),
                .wr_en(ram_wr_en[ram_i]),
                .wr_k(ram_wr_k[ram_i]),
                .wr_data(ram_wr_data[ram_i]),
                .rd_en(ram_rd_en[ram_i]),
                .rd_k(ram_rd_k[ram_i]),
                .rd_data(ram_rd_data[ram_i])
            );
        end
    endgenerate

    always @(*) begin
        launch_slot = launch_m[2:0];

        launch_now = (launch_m < m_size) &&
                     (out_cycle == launch_cycle) &&
                     row_ready[launch_slot] &&
                     !row_stream[launch_slot];

        can_step_out = out_data_ready &&
                       ((launch_m >= m_size) ||
                        (out_cycle != launch_cycle) ||
                        launch_now);

        for (slot = 0; slot < 8; slot = slot + 1) begin
            ram_wr_en[slot] = 4'b0000;
            ram_wr_k[slot] = 64'd0;
            ram_wr_data[slot] = {4*DATA_W{1'b0}};
            ram_rd_en[slot] = 1'b0;
            ram_rd_k[slot] = 16'd0;
            ram_rd_lane[slot] = 2'd0;
        end

        load_m_tmp = load_m;
        load_k_tmp = load_k;
        if (in_data_valid && in_data_ready) begin
            for (lane = 0; lane < 4; lane = lane + 1) begin
                if (load_m_tmp < m_size) begin
                    load_slot = load_m_tmp[2:0];
                    ram_wr_en[load_slot][lane] = 1'b1;
                    ram_wr_k[load_slot][lane*16 +: 16] = load_k_tmp;
                    ram_wr_data[load_slot][lane*DATA_W +: DATA_W] =
                        in_data[lane*DATA_W +: DATA_W];

                    if (load_k_tmp == (k_size - 1)) begin
                        load_k_tmp = 0;
                        load_m_tmp = load_m_tmp + 1;
                    end else begin
                        load_k_tmp = load_k_tmp + 1;
                    end
                end
            end
        end

        if (can_step_out) begin
            if (launch_now) begin
                ram_rd_en[launch_slot] = 1'b1;
                ram_rd_k[launch_slot] = 16'd0;
                ram_rd_lane[launch_slot] = start_lane;
            end

            for (slot = 0; slot < 8; slot = slot + 1) begin
                if (row_stream[slot]) begin
                    ram_rd_en[slot] = 1'b1;
                    ram_rd_k[slot] = row_k_out[slot];
                    ram_rd_lane[slot] = row_start_lane[slot] + row_k_out[slot][1:0];
                end
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
            launch_m <= 0;
            launch_group_cycle <= 0;
            launch_cycle <= 0;
            start_lane <= 0;
            out_cycle <= 0;

            for (slot = 0; slot < 8; slot = slot + 1) begin
                row_ready[slot] <= 1'b0;
                row_stream[slot] <= 1'b0;
                row_m[slot] <= 0;
                row_k_out[slot] <= 0;
                row_start_lane[slot] <= 0;
                rd_valid_q[slot] <= 1'b0;
                rd_lane_q[slot] <= 0;
            end

        end else if (restart) begin
            state <= S_RUN;
            in_data_ready <= 1'b0;
            out_data <= {4*DATA_W{1'b0}};
            out_data_valid <= 1'b0;
            done <= 1'b0;
            load_m <= 0;
            load_k <= 0;
            launch_m <= 0;
            launch_group_cycle <= 0;
            launch_cycle <= 0;
            start_lane <= 0;
            out_cycle <= 0;

            for (slot = 0; slot < 8; slot = slot + 1) begin
                row_ready[slot] <= 1'b0;
                row_stream[slot] <= 1'b0;
                row_m[slot] <= 0;
                row_k_out[slot] <= 0;
                row_start_lane[slot] <= 0;
                rd_valid_q[slot] <= 1'b0;
                rd_lane_q[slot] <= 0;
            end
        end else begin
            case (state)
                S_IDLE: begin
                    in_data_ready <= 1'b0;
                    out_data <= {4*DATA_W{1'b0}};
                    out_data_valid <= 1'b0;
                    done <= 1'b0;
                    load_m <= 0;
                    load_k <= 0;
                    launch_m <= 0;
                    launch_group_cycle <= 0;
                    launch_cycle <= 0;
                    start_lane <= 0;
                    out_cycle <= 0;

                    for (slot = 0; slot < 8; slot = slot + 1) begin
                        row_ready[slot] <= 1'b0;
                        row_stream[slot] <= 1'b0;
                        row_m[slot] <= 0;
                        row_k_out[slot] <= 0;
                        row_start_lane[slot] <= 0;
                        rd_valid_q[slot] <= 1'b0;
                        rd_lane_q[slot] <= 0;
                    end

                    state <= S_IDLE;
                end

                S_RUN: begin
                    in_data_ready <= (load_m < m_size);

                    if (in_data_valid && in_data_ready) begin
                        load_m_tmp = load_m;
                        load_k_tmp = load_k;

                        for (lane = 0; lane < 4; lane = lane + 1) begin
                            if (load_m_tmp < m_size) begin
                                load_slot = load_m_tmp[2:0];
                                row_m[load_slot] <= load_m_tmp;

                                if (load_k_tmp == (k_size - 1)) begin
                                    row_ready[load_slot] <= 1'b1;
                                    load_k_tmp = 0;
                                    load_m_tmp = load_m_tmp + 1;
                                end else begin
                                    load_k_tmp = load_k_tmp + 1;
                                end
                            end
                        end

                        load_m <= load_m_tmp;
                        load_k <= load_k_tmp;
                    end

                    if (can_step_out) begin
                        out_data <= {4*DATA_W{1'b0}};
                        out_data_valid <= 1'b0;
                        any_stream_next = 1'b0;

                        for (slot = 0; slot < 8; slot = slot + 1) begin
                            if (rd_valid_q[slot]) begin
                                out_data[rd_lane_q[slot]*DATA_W +: DATA_W] <=
                                    ram_rd_data[slot];
                                out_data_valid <= 1'b1;
                            end
                        end

                        if (launch_now) begin
                            row_stream[launch_slot] <= 1'b1;
                            row_ready[launch_slot] <= 1'b0;
                            row_k_out[launch_slot] <= 1;
                            row_start_lane[launch_slot] <= start_lane;
                            launch_m <= launch_m + 1;

                            if (launch_m[1:0] == 2'd3) begin
                                launch_group_cycle <= launch_group_cycle + k_size;
                                launch_cycle <= launch_group_cycle + k_size;
                                start_lane <= start_lane + k_size[1:0];
                            end else begin
                                launch_cycle <= launch_cycle + 1;
                            end

                            if (k_size != 1)
                                any_stream_next = 1'b1;
                        end

                        for (slot = 0; slot < 8; slot = slot + 1) begin
                            if (row_stream[slot]) begin
                                stream_slot = slot[2:0];

                                if (row_k_out[stream_slot] == (k_size - 1)) begin
                                    row_stream[stream_slot] <= 1'b0;
                                end else begin
                                    row_k_out[stream_slot] <= row_k_out[stream_slot] + 1;
                                    any_stream_next = 1'b1;
                                end
                            end
                        end

                        out_cycle <= out_cycle + 1;

                        for (slot = 0; slot < 8; slot = slot + 1) begin
                            rd_valid_q[slot] <= ram_rd_en[slot];
                            rd_lane_q[slot] <= ram_rd_lane[slot];
                        end

                        if ((launch_m >= m_size) && !any_stream_next &&
                            (load_m >= m_size) &&
                            !rd_valid_q[0] && !rd_valid_q[1] &&
                            !rd_valid_q[2] && !rd_valid_q[3] &&
                            !rd_valid_q[4] && !rd_valid_q[5] &&
                            !rd_valid_q[6] && !rd_valid_q[7] &&
                            !ram_rd_en[0] && !ram_rd_en[1] &&
                            !ram_rd_en[2] && !ram_rd_en[3] &&
                            !ram_rd_en[4] && !ram_rd_en[5] &&
                            !ram_rd_en[6] && !ram_rd_en[7]) begin
                            done <= 1'b1;
                            in_data_ready <= 1'b0;
                            state <= S_DONE;
                        end
                    end else begin
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
