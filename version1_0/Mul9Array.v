`timescale 1ns / 1ps

module Mul9Array (
    input  wire signed [8:0] a0, a1, a2,
    input  wire signed [8:0] a3, a4, a5,
    input  wire signed [8:0] a6, a7, a8,
    input  wire signed [8:0] b0, b1, b2,
    input  wire signed [8:0] b3, b4, b5,
    input  wire signed [8:0] b6, b7, b8,
    output wire signed [17:0] p0, p1, p2,
    output wire signed [17:0] p3, p4, p5,
    output wire signed [17:0] p6, p7, p8
);
    assign p0 = a0 * b0;
    assign p1 = a1 * b1;
    assign p2 = a2 * b2;
    assign p3 = a3 * b3;
    assign p4 = a4 * b4;
    assign p5 = a5 * b5;
    assign p6 = a6 * b6;
    assign p7 = a7 * b7;
    assign p8 = a8 * b8;
endmodule
