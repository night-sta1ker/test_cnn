`timescale 1ns / 1ps

module Accelerator #(
    parameter IMG_W = 28,
    parameter IMG_H = 28,
    parameter OUT_W = 26,
    parameter OUT_H = 26,
    parameter N_OC  = 16,
    parameter DATA_W = 8,
    parameter ACC_W  = 32
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start,

    input  wire [7:0]                   act,
    input  wire                         act_en,
    output wire                         act_ready,

    input  wire [7:0]                   wgt0, wgt1, wgt2, wgt3,
    input  wire [7:0]                   wgt4, wgt5, wgt6, wgt7,
    input  wire                         wgt_en,
    output wire                         wgt_ready,

    input  wire signed [15:0]           mult0, mult1, mult2, mult3,
    input  wire signed [15:0]           mult4, mult5, mult6, mult7,
    input  wire signed [7:0]            shift_param0, shift_param1,
    input  wire signed [7:0]            shift_param2, shift_param3,
    input  wire signed [7:0]            shift_param4, shift_param5,
    input  wire signed [7:0]            shift_param6, shift_param7,

    output wire [63:0]                  out_parallel,
    output wire                         out_en,
    output wire                         done
);
    localparam OC_GROUPS  = (N_OC + 7) / 8;
    localparam OC_GROUP_W = (OC_GROUPS <= 1) ? 1 :
                            (OC_GROUPS <= 2) ? 1 :
                            (OC_GROUPS <= 4) ? 2 :
                            (OC_GROUPS <= 8) ? 3 : 4;

    reg [OC_GROUP_W-1:0] q_group_cnt;
    reg [3:0] q_k_cnt;
    reg signed [15:0] mult_reg [0:N_OC-1];
    reg signed [7:0]  shft_reg [0:N_OC-1];

    wire [8*ACC_W-1:0] acc_parallel;
    wire acc_en;
    wire [4:0] out_dbg_row;
    wire [4:0] out_dbg_col_base;
    wire [3:0] out_dbg_oc_group;
    wire [2:0] out_dbg_lane;
    wire [4:0] quant_oc_base = out_dbg_oc_group * 5'd8;

    ProcessUnitMode0 #(
        .IMG_W(IMG_W),
        .IMG_H(IMG_H),
        .OUT_W(OUT_W),
        .OUT_H(OUT_H),
        .N_OC(N_OC),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) process_unit_u (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .act(act),
        .act_en(act_en),
        .act_ready(act_ready),
        .wgt0(wgt0), .wgt1(wgt1), .wgt2(wgt2), .wgt3(wgt3),
        .wgt4(wgt4), .wgt5(wgt5), .wgt6(wgt6), .wgt7(wgt7),
        .wgt_en(wgt_en),
        .wgt_ready(wgt_ready),
        .acc_parallel(acc_parallel),
        .acc_en(acc_en),
        .out_dbg_row(out_dbg_row),
        .out_dbg_col_base(out_dbg_col_base),
        .out_dbg_oc_group(out_dbg_oc_group),
        .out_dbg_lane(out_dbg_lane),
        .done(done)
    );

    RequantLane8 #(
        .ACC_W(ACC_W)
    ) requant_u (
        .acc_parallel(acc_parallel),
        .mult0(mult_reg[quant_oc_base + 0]),
        .mult1(mult_reg[quant_oc_base + 1]),
        .mult2(mult_reg[quant_oc_base + 2]),
        .mult3(mult_reg[quant_oc_base + 3]),
        .mult4(mult_reg[quant_oc_base + 4]),
        .mult5(mult_reg[quant_oc_base + 5]),
        .mult6(mult_reg[quant_oc_base + 6]),
        .mult7(mult_reg[quant_oc_base + 7]),
        .shift0(shft_reg[quant_oc_base + 0]),
        .shift1(shft_reg[quant_oc_base + 1]),
        .shift2(shft_reg[quant_oc_base + 2]),
        .shift3(shft_reg[quant_oc_base + 3]),
        .shift4(shft_reg[quant_oc_base + 4]),
        .shift5(shft_reg[quant_oc_base + 5]),
        .shift6(shft_reg[quant_oc_base + 6]),
        .shift7(shft_reg[quant_oc_base + 7]),
        .out_parallel(out_parallel)
    );

    assign out_en = acc_en;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_group_cnt <= {OC_GROUP_W{1'b0}};
            q_k_cnt <= 4'd0;
        end else begin
            if (start) begin
                q_group_cnt <= {OC_GROUP_W{1'b0}};
                q_k_cnt <= 4'd0;
            end else if (wgt_en && wgt_ready) begin
                if (q_k_cnt == 4'd8) begin
                    mult_reg[q_group_cnt * 5'd8 + 0] <= mult0;
                    mult_reg[q_group_cnt * 5'd8 + 1] <= mult1;
                    mult_reg[q_group_cnt * 5'd8 + 2] <= mult2;
                    mult_reg[q_group_cnt * 5'd8 + 3] <= mult3;
                    mult_reg[q_group_cnt * 5'd8 + 4] <= mult4;
                    mult_reg[q_group_cnt * 5'd8 + 5] <= mult5;
                    mult_reg[q_group_cnt * 5'd8 + 6] <= mult6;
                    mult_reg[q_group_cnt * 5'd8 + 7] <= mult7;
                    shft_reg[q_group_cnt * 5'd8 + 0] <= shift_param0;
                    shft_reg[q_group_cnt * 5'd8 + 1] <= shift_param1;
                    shft_reg[q_group_cnt * 5'd8 + 2] <= shift_param2;
                    shft_reg[q_group_cnt * 5'd8 + 3] <= shift_param3;
                    shft_reg[q_group_cnt * 5'd8 + 4] <= shift_param4;
                    shft_reg[q_group_cnt * 5'd8 + 5] <= shift_param5;
                    shft_reg[q_group_cnt * 5'd8 + 6] <= shift_param6;
                    shft_reg[q_group_cnt * 5'd8 + 7] <= shift_param7;
                    q_k_cnt <= 4'd0;
                    if (q_group_cnt != OC_GROUPS-1)
                        q_group_cnt <= q_group_cnt + 1'b1;
                end else begin
                    q_k_cnt <= q_k_cnt + 4'd1;
                end
            end
        end
    end
endmodule
