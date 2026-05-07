`timescale 1ns / 1ps

// 3x3 dot-product PE: acc = sum(act[i] * wgt[i]) for i=0..8
// act: unsigned 8-bit, wgt: signed 8-bit, acc: signed 32-bit
module PE (
    input  wire clk,
    input  wire rst_n,
    input  wire [7:0]  act0, act1, act2,
    input  wire [7:0]  act3, act4, act5,
    input  wire [7:0]  act6, act7, act8,
    input  wire signed [7:0]  wgt0, wgt1, wgt2,
    input  wire signed [7:0]  wgt3, wgt4, wgt5,
    input  wire signed [7:0]  wgt6, wgt7, wgt8,
    output reg signed [31:0] acc
);
    wire signed [15:0] p0 = $signed({1'b0, act0}) * wgt0;
    wire signed [15:0] p1 = $signed({1'b0, act1}) * wgt1;
    wire signed [15:0] p2 = $signed({1'b0, act2}) * wgt2;
    wire signed [15:0] p3 = $signed({1'b0, act3}) * wgt3;
    wire signed [15:0] p4 = $signed({1'b0, act4}) * wgt4;
    wire signed [15:0] p5 = $signed({1'b0, act5}) * wgt5;
    wire signed [15:0] p6 = $signed({1'b0, act6}) * wgt6;
    wire signed [15:0] p7 = $signed({1'b0, act7}) * wgt7;
    wire signed [15:0] p8 = $signed({1'b0, act8}) * wgt8;

    wire signed [31:0] p0_ext = {{16{p0[15]}}, p0};
    wire signed [31:0] p1_ext = {{16{p1[15]}}, p1};
    wire signed [31:0] p2_ext = {{16{p2[15]}}, p2};
    wire signed [31:0] p3_ext = {{16{p3[15]}}, p3};
    wire signed [31:0] p4_ext = {{16{p4[15]}}, p4};
    wire signed [31:0] p5_ext = {{16{p5[15]}}, p5};
    wire signed [31:0] p6_ext = {{16{p6[15]}}, p6};
    wire signed [31:0] p7_ext = {{16{p7[15]}}, p7};
    wire signed [31:0] p8_ext = {{16{p8[15]}}, p8};

    wire signed [31:0] sum01 = p0_ext + p1_ext;
    wire signed [31:0] sum23 = p2_ext + p3_ext;
    wire signed [31:0] sum45 = p4_ext + p5_ext;
    wire signed [31:0] sum67 = p6_ext + p7_ext;
    wire signed [31:0] sum03 = sum01 + sum23;
    wire signed [31:0] sum47 = sum45 + sum67;
    wire signed [31:0] acc_comb = sum03 + sum47 + p8_ext;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc <= 32'sd0;
        else
            acc <= acc_comb;
    end
endmodule
