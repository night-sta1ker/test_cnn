`timescale 1ns / 1ps

module RequantU8 #(
    parameter ACC_W = 32
) (
    input  wire signed [ACC_W-1:0] acc_in,
    input  wire signed [15:0]      mult_in,
    input  wire signed [7:0]       shift_in,
    output reg  [7:0]              q_out
);
    reg signed [47:0] prod;
    reg signed [7:0]  q_sum;
    reg [5:0]         shamt;
    reg signed [47:0] qv;

    always @(*) begin
        prod = acc_in * mult_in;
        q_sum = 8'sd15 + shift_in;
        shamt = (q_sum > 8'sd63) ? 6'd63 :
                (q_sum < 8'sd0)  ? 6'd0  : q_sum[5:0];
        qv = prod >>> shamt;
        if (qv < 0)
            q_out = 8'd0;
        else if (qv > 255)
            q_out = 8'd255;
        else
            q_out = qv[7:0];
    end
endmodule

module RequantLane8 #(
    parameter ACC_W = 32
) (
    input  wire signed [8*ACC_W-1:0] acc_parallel,
    input  wire signed [15:0]        mult0, mult1, mult2, mult3,
    input  wire signed [15:0]        mult4, mult5, mult6, mult7,
    input  wire signed [7:0]         shift0, shift1, shift2, shift3,
    input  wire signed [7:0]         shift4, shift5, shift6, shift7,
    output wire [63:0]               out_parallel
);
    wire [7:0] q0, q1, q2, q3;
    wire [7:0] q4, q5, q6, q7;

    RequantU8 #(.ACC_W(ACC_W)) q_u0 (.acc_in(acc_parallel[0*ACC_W +: ACC_W]), .mult_in(mult0), .shift_in(shift0), .q_out(q0));
    RequantU8 #(.ACC_W(ACC_W)) q_u1 (.acc_in(acc_parallel[1*ACC_W +: ACC_W]), .mult_in(mult1), .shift_in(shift1), .q_out(q1));
    RequantU8 #(.ACC_W(ACC_W)) q_u2 (.acc_in(acc_parallel[2*ACC_W +: ACC_W]), .mult_in(mult2), .shift_in(shift2), .q_out(q2));
    RequantU8 #(.ACC_W(ACC_W)) q_u3 (.acc_in(acc_parallel[3*ACC_W +: ACC_W]), .mult_in(mult3), .shift_in(shift3), .q_out(q3));
    RequantU8 #(.ACC_W(ACC_W)) q_u4 (.acc_in(acc_parallel[4*ACC_W +: ACC_W]), .mult_in(mult4), .shift_in(shift4), .q_out(q4));
    RequantU8 #(.ACC_W(ACC_W)) q_u5 (.acc_in(acc_parallel[5*ACC_W +: ACC_W]), .mult_in(mult5), .shift_in(shift5), .q_out(q5));
    RequantU8 #(.ACC_W(ACC_W)) q_u6 (.acc_in(acc_parallel[6*ACC_W +: ACC_W]), .mult_in(mult6), .shift_in(shift6), .q_out(q6));
    RequantU8 #(.ACC_W(ACC_W)) q_u7 (.acc_in(acc_parallel[7*ACC_W +: ACC_W]), .mult_in(mult7), .shift_in(shift7), .q_out(q7));

    assign out_parallel = {q7, q6, q5, q4, q3, q2, q1, q0};
endmodule
