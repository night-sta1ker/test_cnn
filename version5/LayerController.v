`timescale 1ns / 1ps

// Fixed inference_pure layer schedule. gemm_done must include result-writer drain.
module LayerController (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire        gemm_done,
    input  wire        pool_done,
    input  wire        argmax_done,

    output wire        busy,
    output reg         done,
    output reg         gemm_start,
    output reg         pool_start,
    output reg         argmax_start,
    output reg  [2:0]  stage,

    output reg         act_mode,
    output reg  [15:0] act_h,
    output reg  [15:0] act_w,
    output reg  [2:0]  act_channels,
    output reg  [15:0] act_channel_stride,
    output reg  [15:0] act_vector_size,
    output reg  [11:0] act_base_byte,

    output reg  [15:0] seq_m,
    output reg  [15:0] seq_k,
    output reg  [15:0] seq_n,
    output reg  [9:0]  wgt_base_word,

    output reg  [11:0] result_base_byte,
    output reg  [5:0]  result_out_w,
    output reg  [5:0]  result_row_stride,
    output reg  [11:0] result_plane_size,
    output reg  [4:0]  param_base,

    output reg  [11:0] pool_src_base_byte,
    output reg  [11:0] pool_dst_base_byte,
    output reg  [3:0]  pool_channels,
    output reg  [5:0]  pool_h,
    output reg  [5:0]  pool_w,
    output reg  [11:0] pool_plane_stride,

    output reg  [11:0] argmax_src_base_byte
);
    localparam ST_IDLE  = 3'd0;
    localparam ST_CONV1 = 3'd1;
    localparam ST_POOL1 = 3'd2;
    localparam ST_CONV2 = 3'd3;
    localparam ST_POOL2 = 3'd4;
    localparam ST_FC    = 3'd5;
    localparam ST_ARG   = 3'd6;

    assign busy = (stage != ST_IDLE);

    always @* begin
        act_mode = 1'b0;
        act_h = 0;
        act_w = 0;
        act_channels = 0;
        act_channel_stride = 0;
        act_vector_size = 0;
        act_base_byte = 0;
        seq_m = 0;
        seq_k = 0;
        seq_n = 0;
        wgt_base_word = 0;
        result_base_byte = 12'd1024;
        result_out_w = 0;
        result_row_stride = 0;
        result_plane_size = 0;
        param_base = 0;
        pool_src_base_byte = 12'd1024;
        pool_dst_base_byte = 0;
        pool_channels = 0;
        pool_h = 0;
        pool_w = 0;
        pool_plane_stride = 0;
        argmax_src_base_byte = 12'd1024;

        case (stage)
            ST_CONV1: begin
                act_h = 16'd28;
                act_w = 16'd28;
                act_channels = 3'd1;
                act_channel_stride = 16'd784;
                seq_m = 16'd728;
                seq_k = 16'd9;
                seq_n = 16'd4;
                wgt_base_word = 10'd0;
                result_out_w = 6'd26;
                result_row_stride = 6'd28;
                result_plane_size = 12'd676;
                param_base = 5'd0;
            end
            ST_POOL1: begin
                pool_channels = 4'd4;
                pool_h = 6'd26;
                pool_w = 6'd26;
                pool_plane_stride = 12'd676;
            end
            ST_CONV2: begin
                act_h = 16'd13;
                act_w = 16'd13;
                act_channels = 3'd4;
                act_channel_stride = 16'd169;
                seq_m = 16'd132;
                seq_k = 16'd36;
                seq_n = 16'd8;
                wgt_base_word = 10'd9;
                result_out_w = 6'd11;
                result_row_stride = 6'd12;
                result_plane_size = 12'd121;
                param_base = 5'd4;
            end
            ST_POOL2: begin
                pool_channels = 4'd8;
                pool_h = 6'd11;
                pool_w = 6'd11;
                pool_plane_stride = 12'd121;
            end
            ST_FC: begin
                act_mode = 1'b1;
                act_vector_size = 16'd200;
                seq_m = 16'd1;
                seq_k = 16'd200;
                seq_n = 16'd10;
                wgt_base_word = 10'd81;
                result_out_w = 6'd1;
                result_row_stride = 6'd1;
                result_plane_size = 12'd1;
                param_base = 5'd12;
            end
            default: begin
            end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage <= ST_IDLE;
            done <= 1'b0;
            gemm_start <= 1'b0;
            pool_start <= 1'b0;
            argmax_start <= 1'b0;
        end
        else begin
            done <= 1'b0;
            gemm_start <= 1'b0;
            pool_start <= 1'b0;
            argmax_start <= 1'b0;
            case (stage)
                ST_IDLE: if (start) begin
                    stage <= ST_CONV1;
                    gemm_start <= 1'b1;
                end
                ST_CONV1: if (gemm_done) begin
                    stage <= ST_POOL1;
                    pool_start <= 1'b1;
                end
                ST_POOL1: if (pool_done) begin
                    stage <= ST_CONV2;
                    gemm_start <= 1'b1;
                end
                ST_CONV2: if (gemm_done) begin
                    stage <= ST_POOL2;
                    pool_start <= 1'b1;
                end
                ST_POOL2: if (pool_done) begin
                    stage <= ST_FC;
                    gemm_start <= 1'b1;
                end
                ST_FC: if (gemm_done) begin
                    stage <= ST_ARG;
                    argmax_start <= 1'b1;
                end
                ST_ARG: if (argmax_done) begin
                    stage <= ST_IDLE;
                    done <= 1'b1;
                end
                default: stage <= ST_IDLE;
            endcase
        end
    end
endmodule
