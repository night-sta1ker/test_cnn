`timescale 1ns / 1ps

// mode=0: 3x3 MAC, acc <= sum(act[i] * wgt[i])
// mode=1: reuse the 9x9 multiplier array for acc * quant_mult.
module PE (
    input  wire clk,
    input  wire rst_n,
    input  wire mode,

    input  wire [7:0]  act0, act1, act2,
    input  wire [7:0]  act3, act4, act5,
    input  wire [7:0]  act6, act7, act8,
    input  wire signed [7:0]  wgt0, wgt1, wgt2,
    input  wire signed [7:0]  wgt3, wgt4, wgt5,
    input  wire signed [7:0]  wgt6, wgt7, wgt8,

    input  wire signed [15:0] quant_mult,
    output reg  signed [31:0] acc,
    output wire signed [47:0] quant_prod
);
    localparam MODE_MAC   = 1'b0;
    localparam MODE_QUANT = 1'b1;

    wire acc_neg  = acc[31];
    wire mult_neg = quant_mult[15];
    wire prod_neg = acc_neg ^ mult_neg;

    wire [31:0] acc_abs  = acc_neg  ? (~acc + 32'd1) : acc;
    wire [15:0] mult_abs = mult_neg ? (~quant_mult + 16'd1) : quant_mult;

    wire [7:0] acc_b0 = acc_abs[7:0];
    wire [7:0] acc_b1 = acc_abs[15:8];
    wire [7:0] acc_b2 = acc_abs[23:16];
    wire [7:0] acc_b3 = acc_abs[31:24];
    wire [7:0] mul_b0 = mult_abs[7:0];
    wire [7:0] mul_b1 = mult_abs[15:8];

    wire signed [8:0] arr_a0 = (mode == MODE_MAC) ? $signed({1'b0, act0}) : $signed({1'b0, acc_b0});
    wire signed [8:0] arr_a1 = (mode == MODE_MAC) ? $signed({1'b0, act1}) : $signed({1'b0, acc_b1});
    wire signed [8:0] arr_a2 = (mode == MODE_MAC) ? $signed({1'b0, act2}) : $signed({1'b0, acc_b2});
    wire signed [8:0] arr_a3 = (mode == MODE_MAC) ? $signed({1'b0, act3}) : $signed({1'b0, acc_b3});
    wire signed [8:0] arr_a4 = (mode == MODE_MAC) ? $signed({1'b0, act4}) : $signed({1'b0, acc_b0});
    wire signed [8:0] arr_a5 = (mode == MODE_MAC) ? $signed({1'b0, act5}) : $signed({1'b0, acc_b1});
    wire signed [8:0] arr_a6 = (mode == MODE_MAC) ? $signed({1'b0, act6}) : $signed({1'b0, acc_b2});
    wire signed [8:0] arr_a7 = (mode == MODE_MAC) ? $signed({1'b0, act7}) : $signed({1'b0, acc_b3});
    wire signed [8:0] arr_a8 = (mode == MODE_MAC) ? $signed({1'b0, act8}) : 9'sd0;

    wire signed [8:0] arr_b0 = (mode == MODE_MAC) ? {wgt0[7], wgt0} : $signed({1'b0, mul_b0});
    wire signed [8:0] arr_b1 = (mode == MODE_MAC) ? {wgt1[7], wgt1} : $signed({1'b0, mul_b0});
    wire signed [8:0] arr_b2 = (mode == MODE_MAC) ? {wgt2[7], wgt2} : $signed({1'b0, mul_b0});
    wire signed [8:0] arr_b3 = (mode == MODE_MAC) ? {wgt3[7], wgt3} : $signed({1'b0, mul_b0});
    wire signed [8:0] arr_b4 = (mode == MODE_MAC) ? {wgt4[7], wgt4} : $signed({1'b0, mul_b1});
    wire signed [8:0] arr_b5 = (mode == MODE_MAC) ? {wgt5[7], wgt5} : $signed({1'b0, mul_b1});
    wire signed [8:0] arr_b6 = (mode == MODE_MAC) ? {wgt6[7], wgt6} : $signed({1'b0, mul_b1});
    wire signed [8:0] arr_b7 = (mode == MODE_MAC) ? {wgt7[7], wgt7} : $signed({1'b0, mul_b1});
    wire signed [8:0] arr_b8 = (mode == MODE_MAC) ? {wgt8[7], wgt8} : 9'sd0;

    wire signed [17:0] p0, p1, p2, p3, p4, p5, p6, p7, p8;

    Mul9Array mul_u (
        .a0(arr_a0), .a1(arr_a1), .a2(arr_a2),
        .a3(arr_a3), .a4(arr_a4), .a5(arr_a5),
        .a6(arr_a6), .a7(arr_a7), .a8(arr_a8),
        .b0(arr_b0), .b1(arr_b1), .b2(arr_b2),
        .b3(arr_b3), .b4(arr_b4), .b5(arr_b5),
        .b6(arr_b6), .b7(arr_b7), .b8(arr_b8),
        .p0(p0), .p1(p1), .p2(p2),
        .p3(p3), .p4(p4), .p5(p5),
        .p6(p6), .p7(p7), .p8(p8)
    );

    wire signed [31:0] p0_ext = {{14{p0[17]}}, p0};
    wire signed [31:0] p1_ext = {{14{p1[17]}}, p1};
    wire signed [31:0] p2_ext = {{14{p2[17]}}, p2};
    wire signed [31:0] p3_ext = {{14{p3[17]}}, p3};
    wire signed [31:0] p4_ext = {{14{p4[17]}}, p4};
    wire signed [31:0] p5_ext = {{14{p5[17]}}, p5};
    wire signed [31:0] p6_ext = {{14{p6[17]}}, p6};
    wire signed [31:0] p7_ext = {{14{p7[17]}}, p7};
    wire signed [31:0] p8_ext = {{14{p8[17]}}, p8};

    wire signed [31:0] sum01 = p0_ext + p1_ext;
    wire signed [31:0] sum23 = p2_ext + p3_ext;
    wire signed [31:0] sum45 = p4_ext + p5_ext;
    wire signed [31:0] sum67 = p6_ext + p7_ext;
    wire signed [31:0] sum03 = sum01 + sum23;
    wire signed [31:0] sum47 = sum45 + sum67;
    wire signed [31:0] acc_comb = sum03 + sum47 + p8_ext;

    wire [47:0] q0 = {30'd0, p0[17:0]};
    wire [47:0] q1 = {30'd0, p1[17:0]} << 8;
    wire [47:0] q2 = {30'd0, p2[17:0]} << 16;
    wire [47:0] q3 = {30'd0, p3[17:0]} << 24;
    wire [47:0] q4 = {30'd0, p4[17:0]} << 8;
    wire [47:0] q5 = {30'd0, p5[17:0]} << 16;
    wire [47:0] q6 = {30'd0, p6[17:0]} << 24;
    wire [47:0] q7 = {30'd0, p7[17:0]} << 32;
    wire [47:0] quant_abs_prod = q0 + q1 + q2 + q3 + q4 + q5 + q6 + q7;

    assign quant_prod = prod_neg ? -$signed(quant_abs_prod) : $signed(quant_abs_prod);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc <= 32'sd0;
        else if (mode == MODE_MAC)
            acc <= acc_comb;
    end
endmodule
