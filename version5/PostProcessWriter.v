`timescale 1ns / 1ps

module PostProcessWriter #(
    parameter SRAM_ADDR_W = 10,
    parameter QUEUE_ADDR_W = 2
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clear,
    input  wire                    cfg_valid,
    input  wire                    cfg_mode_fc,
    input  wire [11:0]             cfg_result_base_byte,
    input  wire [5:0]              cfg_out_w,
    input  wire [5:0]              cfg_row_stride,
    input  wire [11:0]             cfg_plane_size,
    input  wire [15:0]             cfg_m_size,
    input  wire [15:0]             cfg_n_size,

    input  wire [127:0]            psum_data,
    input  wire [3:0]              psum_valid,
    output wire                    psum_ready,
    input  wire [15:0]             result_m0,
    input  wire [15:0]             result_m1,
    input  wire [15:0]             result_m2,
    input  wire [15:0]             result_m3,
    input  wire [15:0]             result_n_base,
    input  wire                    seq_done,

    input  wire [127:0]            param_bias,
    input  wire [63:0]             param_mult,
    input  wire [31:0]             param_shift,
    input  wire [31:0]             param_zout,
    input  wire [3:0]              param_relu,

    output wire                    busy,
    output reg                     done,
    output reg                     overflow,
    output reg                     protocol_error,
    output wire                    sram_wr_en,
    output wire [SRAM_ADDR_W-1:0]  sram_wr_addr,
    output wire [31:0]             sram_wr_data,
    output wire [3:0]              sram_wr_be
);
    localparam QUEUE_DEPTH = (1 << QUEUE_ADDR_W);

    reg active;
    reg mode_fc_r;
    reg [5:0] out_w_r;
    reg [5:0] row_stride_r;
    reg [11:0] plane_size_r;
    reg [15:0] m_size_r;
    reg [15:0] n_size_r;
    reg [15:0] tile_n_base;
    reg seq_done_seen;

    wire [31:0] lane_psum [0:3];
    wire [15:0] lane_m [0:3];
    wire [7:0] lane_q [0:3];
    reg [3:0] q1_valid;
    reg signed [31:0] q1_acc [0:3];
    reg signed [15:0] q1_mult [0:3];
    reg [7:0] q1_shift [0:3];
    reg [7:0] q1_zout [0:3];
    reg q1_relu [0:3];
    reg [15:0] q1_m [0:3];
    reg [15:0] q1_n_base;
    reg [3:0] q2_valid;
    reg signed [47:0] q2_product [0:3];
    reg [7:0] q2_shift [0:3];
    reg [7:0] q2_zout [0:3];
    reg q2_relu [0:3];
    reg [15:0] q2_m [0:3];
    reg [15:0] q2_n_base;
    reg [3:0] q3_valid;
    reg signed [47:0] q3_scaled [0:3];
    reg [7:0] q3_zout [0:3];
    reg q3_relu [0:3];
    reg [15:0] q3_m [0:3];
    reg [15:0] q3_n_base;
    reg [5:0] lane_x [0:3];
    reg [11:0] lane_byte_addr [0:3];
    reg [31:0] assemble_data [0:3];
    reg [3:0] assemble_be [0:3];
    reg [SRAM_ADDR_W-1:0] assemble_addr [0:3];

    reg emit [0:3];
    reg [31:0] emit_data [0:3];
    reg [3:0] emit_be [0:3];
    reg [SRAM_ADDR_W-1:0] emit_addr [0:3];
    reg physical [0:3];
    reg last_m [0:3];
    reg [31:0] append_data [0:3];
    reg [3:0] append_be [0:3];

    reg [SRAM_ADDR_W-1:0] lane_fifo_addr [0:3][0:QUEUE_DEPTH-1];
    reg [31:0] lane_fifo_data [0:3][0:QUEUE_DEPTH-1];
    reg [3:0] lane_fifo_be [0:3][0:QUEUE_DEPTH-1];
    reg [QUEUE_ADDR_W-1:0] lane_wr_ptr [0:3];
    reg [QUEUE_ADDR_W-1:0] lane_rd_ptr [0:3];
    reg [QUEUE_ADDR_W:0] lane_count [0:3];
    reg [1:0] rr_lane;
    reg [1:0] pop_lane;
    reg pop_valid;

    integer arb_offset;
    always @* begin
        pop_lane = rr_lane;
        pop_valid = 1'b0;
        for (arb_offset = 0; arb_offset < 4; arb_offset = arb_offset + 1) begin
            if (!pop_valid &&
                (lane_count[(rr_lane + arb_offset) & 2'b11] != 0)) begin
                pop_lane = (rr_lane + arb_offset) & 2'b11;
                pop_valid = 1'b1;
            end
        end
    end

    wire queue_pop = pop_valid;
    wire queue_nonempty = (lane_count[0] != 0) || (lane_count[1] != 0) ||
                          (lane_count[2] != 0) || (lane_count[3] != 0);
    wire [3:0] lane_can_push;
    genvar queue_lane;
    generate
        for (queue_lane = 0; queue_lane < 4; queue_lane = queue_lane + 1) begin : gen_queue_space
            assign lane_can_push[queue_lane] =
                (lane_count[queue_lane] < QUEUE_DEPTH) ||
                (queue_pop && (pop_lane == queue_lane));
        end
    endgenerate
    wire pipe_advance = (!emit[0] || lane_can_push[0]) &&
                        (!emit[1] || lane_can_push[1]) &&
                        (!emit[2] || lane_can_push[2]) &&
                        (!emit[3] || lane_can_push[3]);
    wire tile_prepare = (q1_valid != 0) && (q1_n_base != tile_n_base) &&
                        (q2_valid == 0) && (q3_valid == 0);
    wire tile_overlap = (q1_valid != 0) && (q1_n_base != tile_n_base) &&
                        ((q2_valid != 0) || (q3_valid != 0));
    wire quant_pipe_empty = (q1_valid == 0) && (q2_valid == 0) &&
                            (q3_valid == 0);
    wire assemblers_empty = (assemble_be[0] == 0) && (assemble_be[1] == 0) &&
                            (assemble_be[2] == 0) && (assemble_be[3] == 0);

    assign busy = active || queue_nonempty;
    assign psum_ready = pipe_advance;
    assign sram_wr_en = queue_pop;
    assign sram_wr_addr = queue_pop ?
        lane_fifo_addr[pop_lane][lane_rd_ptr[pop_lane]] : 0;
    assign sram_wr_data = queue_pop ?
        lane_fifo_data[pop_lane][lane_rd_ptr[pop_lane]] : 0;
    assign sram_wr_be = queue_pop ?
        lane_fifo_be[pop_lane][lane_rd_ptr[pop_lane]] : 0;

    assign lane_psum[0] = psum_data[31:0];
    assign lane_psum[1] = psum_data[63:32];
    assign lane_psum[2] = psum_data[95:64];
    assign lane_psum[3] = psum_data[127:96];
    assign lane_m[0] = result_m0;
    assign lane_m[1] = result_m1;
    assign lane_m[2] = result_m2;
    assign lane_m[3] = result_m3;

    function automatic [7:0] requant_finish;
        input [47:0] scaled;
        input [7:0] zout;
        input relu;
        reg signed [47:0] value;
        begin
            value = $signed(scaled) + $signed({1'b0, zout});
            if (relu && value < $signed({1'b0, zout}))
                value = $signed({1'b0, zout});
            if (value < 0)
                requant_finish = 8'd0;
            else if (value > 255)
                requant_finish = 8'd255;
            else
                requant_finish = value[7:0];
        end
    endfunction

    genvar g;
    generate
        for (g = 0; g < 4; g = g + 1) begin : gen_requant
            assign lane_q[g] = requant_finish(
                q3_scaled[g], q3_zout[g], q3_relu[g]
            );
        end
    endgenerate

    integer pipe_lane;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q1_valid <= 0;
            q1_n_base <= 0;
            q2_valid <= 0;
            q2_n_base <= 0;
            q3_valid <= 0;
            q3_n_base <= 0;
            for (pipe_lane = 0; pipe_lane < 4; pipe_lane = pipe_lane + 1) begin
                q1_acc[pipe_lane] <= 0;
                q1_mult[pipe_lane] <= 0;
                q1_shift[pipe_lane] <= 0;
                q1_zout[pipe_lane] <= 0;
                q1_relu[pipe_lane] <= 0;
                q1_m[pipe_lane] <= 0;
                q2_product[pipe_lane] <= 0;
                q2_shift[pipe_lane] <= 0;
                q2_zout[pipe_lane] <= 0;
                q2_relu[pipe_lane] <= 0;
                q2_m[pipe_lane] <= 0;
                q3_scaled[pipe_lane] <= 0;
                q3_zout[pipe_lane] <= 0;
                q3_relu[pipe_lane] <= 0;
                q3_m[pipe_lane] <= 0;
            end
        end
        else if (clear || cfg_valid) begin
            q1_valid <= 0;
            q2_valid <= 0;
            q3_valid <= 0;
            q1_n_base <= 0;
            q2_n_base <= 0;
            q3_n_base <= 0;
        end
        else if (pipe_advance) begin
            q1_valid <= 0;
            q1_n_base <= result_n_base;
            q2_valid <= q1_valid;
            q2_n_base <= q1_n_base;
            q3_valid <= q2_valid;
            q3_n_base <= q2_n_base;
            for (pipe_lane = 0; pipe_lane < 4; pipe_lane = pipe_lane + 1) begin
                q1_valid[pipe_lane] <= active && psum_valid[pipe_lane] &&
                    (result_n_base + pipe_lane < n_size_r);
                q1_acc[pipe_lane] <= $signed(lane_psum[pipe_lane]) +
                                     $signed(param_bias[pipe_lane*32 +: 32]);
                q1_mult[pipe_lane] <= param_mult[pipe_lane*16 +: 16];
                q1_shift[pipe_lane] <= param_shift[pipe_lane*8 +: 8];
                q1_zout[pipe_lane] <= param_zout[pipe_lane*8 +: 8];
                q1_relu[pipe_lane] <= param_relu[pipe_lane];
                q1_m[pipe_lane] <= lane_m[pipe_lane];
                q2_product[pipe_lane] <= q1_acc[pipe_lane] * q1_mult[pipe_lane];
                q2_shift[pipe_lane] <= q1_shift[pipe_lane];
                q2_zout[pipe_lane] <= q1_zout[pipe_lane];
                q2_relu[pipe_lane] <= q1_relu[pipe_lane];
                q2_m[pipe_lane] <= q1_m[pipe_lane];
                q3_scaled[pipe_lane] <= $signed(q2_product[pipe_lane]) >>>
                                        (15 + q2_shift[pipe_lane]);
                q3_zout[pipe_lane] <= q2_zout[pipe_lane];
                q3_relu[pipe_lane] <= q2_relu[pipe_lane];
                q3_m[pipe_lane] <= q2_m[pipe_lane];
            end
        end
    end

    integer i;
    always @* begin
        for (i = 0; i < 4; i = i + 1) begin
            physical[i] = active && q3_valid[i] &&
                          (q3_n_base + i < n_size_r) &&
                          (mode_fc_r || lane_x[i] < out_w_r);
            last_m[i] = active && q3_valid[i] &&
                        (q3_n_base + i < n_size_r) &&
                        (q3_m[i] + 1'b1 >= m_size_r);
            append_data[i] = assemble_data[i];
            append_data[i][lane_byte_addr[i][1:0]*8 +: 8] = lane_q[i];
            append_be[i] = assemble_be[i] | (4'b0001 << lane_byte_addr[i][1:0]);
            emit[i] = (physical[i] &&
                       ((lane_byte_addr[i][1:0] == 2'd3) || last_m[i])) ||
                      (!physical[i] && last_m[i] && (assemble_be[i] != 0));
            emit_data[i] = physical[i] ? append_data[i] : assemble_data[i];
            emit_be[i] = physical[i] ? append_be[i] : assemble_be[i];
            emit_addr[i] = physical[i] ? lane_byte_addr[i][SRAM_ADDR_W+1:2] :
                                         assemble_addr[i];
        end
    end

    integer lane;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0;
            mode_fc_r <= 1'b0;
            out_w_r <= 0;
            row_stride_r <= 0;
            plane_size_r <= 0;
            m_size_r <= 0;
            n_size_r <= 0;
            tile_n_base <= 0;
            seq_done_seen <= 1'b0;
            done <= 1'b0;
            overflow <= 1'b0;
            protocol_error <= 1'b0;
            rr_lane <= 0;
            for (lane = 0; lane < 4; lane = lane + 1) begin
                lane_wr_ptr[lane] <= 0;
                lane_rd_ptr[lane] <= 0;
                lane_count[lane] <= 0;
                lane_x[lane] <= 0;
                lane_byte_addr[lane] <= 0;
                assemble_data[lane] <= 0;
                assemble_be[lane] <= 0;
                assemble_addr[lane] <= 0;
            end
        end
        else if (clear) begin
            active <= 1'b0;
            seq_done_seen <= 1'b0;
            done <= 1'b0;
            overflow <= 1'b0;
            protocol_error <= 1'b0;
            rr_lane <= 0;
            for (lane = 0; lane < 4; lane = lane + 1) begin
                lane_wr_ptr[lane] <= 0;
                lane_rd_ptr[lane] <= 0;
                lane_count[lane] <= 0;
                assemble_be[lane] <= 0;
            end
        end
        else begin
            done <= 1'b0;

            if (cfg_valid) begin
                active <= 1'b1;
                mode_fc_r <= cfg_mode_fc;
                out_w_r <= cfg_out_w;
                row_stride_r <= cfg_row_stride;
                plane_size_r <= cfg_plane_size;
                m_size_r <= cfg_m_size;
                n_size_r <= cfg_n_size;
                tile_n_base <= 0;
                seq_done_seen <= 1'b0;
                overflow <= 1'b0;
                protocol_error <= busy;
                rr_lane <= 0;
                lane_byte_addr[0] <= cfg_result_base_byte;
                lane_byte_addr[1] <= cfg_result_base_byte + cfg_plane_size;
                lane_byte_addr[2] <= cfg_result_base_byte + (cfg_plane_size << 1);
                lane_byte_addr[3] <= cfg_result_base_byte + cfg_plane_size +
                                     (cfg_plane_size << 1);
                for (lane = 0; lane < 4; lane = lane + 1) begin
                    lane_wr_ptr[lane] <= 0;
                    lane_rd_ptr[lane] <= 0;
                    lane_count[lane] <= 0;
                    lane_x[lane] <= 0;
                    assemble_data[lane] <= 0;
                    assemble_be[lane] <= 0;
                    assemble_addr[lane] <= 0;
                end
            end
            else begin
                if (seq_done)
                    seq_done_seen <= 1'b1;

                if (pipe_advance) begin
                    if (tile_prepare) begin
                        if (!assemblers_empty)
                            protocol_error <= 1'b1;
                        tile_n_base <= q1_n_base;
                        for (lane = 0; lane < 4; lane = lane + 1) begin
                            lane_x[lane] <= 0;
                            lane_byte_addr[lane] <= lane_byte_addr[lane] +
                                                    plane_size_r +
                                                    (plane_size_r << 1);
                            assemble_data[lane] <= 0;
                            assemble_be[lane] <= 0;
                        end
                    end
                    else begin
                        for (lane = 0; lane < 4; lane = lane + 1) begin
                            if (active && q3_valid[lane] &&
                                (q3_n_base + lane < n_size_r)) begin
                                if (lane_x[lane] + 1'b1 >= row_stride_r)
                                    lane_x[lane] <= 0;
                                else
                                    lane_x[lane] <= lane_x[lane] + 1'b1;

                                if (physical[lane]) begin
                                    lane_byte_addr[lane] <= lane_byte_addr[lane] + 1'b1;
                                    if (emit[lane]) begin
                                        assemble_data[lane] <= 0;
                                        assemble_be[lane] <= 0;
                                    end
                                    else begin
                                        assemble_data[lane] <= append_data[lane];
                                        assemble_be[lane] <= append_be[lane];
                                        assemble_addr[lane] <=
                                            lane_byte_addr[lane][SRAM_ADDR_W+1:2];
                                    end
                                end
                                else if (emit[lane]) begin
                                    assemble_data[lane] <= 0;
                                    assemble_be[lane] <= 0;
                                end
                            end
                        end
                    end

                    if (tile_overlap)
                        protocol_error <= 1'b1;

                    for (lane = 0; lane < 4; lane = lane + 1) begin
                        if (emit[lane]) begin
                            lane_fifo_addr[lane][lane_wr_ptr[lane]] <= emit_addr[lane];
                            lane_fifo_data[lane][lane_wr_ptr[lane]] <= emit_data[lane];
                            lane_fifo_be[lane][lane_wr_ptr[lane]] <= emit_be[lane];
                            lane_wr_ptr[lane] <= lane_wr_ptr[lane] + 1'b1;
                        end
                    end
                end

                if (queue_pop) begin
                    lane_rd_ptr[pop_lane] <= lane_rd_ptr[pop_lane] + 1'b1;
                    rr_lane <= pop_lane + 1'b1;
                end

                for (lane = 0; lane < 4; lane = lane + 1) begin
                    case ({pipe_advance && emit[lane],
                           queue_pop && (pop_lane == lane)})
                        2'b10: lane_count[lane] <= lane_count[lane] + 1'b1;
                        2'b01: lane_count[lane] <= lane_count[lane] - 1'b1;
                        default: begin
                        end
                    endcase
                    if (lane_count[lane] > QUEUE_DEPTH)
                        overflow <= 1'b1;
                end

                if (active && (seq_done_seen || seq_done) &&
                    !queue_nonempty &&
                    assemblers_empty && quant_pipe_empty &&
                    (psum_valid == 0)) begin
                    active <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end
endmodule
