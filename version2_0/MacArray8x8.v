`timescale 1ns / 1ps

module MacCell #(
    parameter DATA_W = 8,
    parameter ACC_W  = 32
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         clear,
    input  wire                         en,
    input  wire        [DATA_W-1:0]     a,
    input  wire signed [DATA_W-1:0]     b,
    output reg  signed [ACC_W-1:0]      acc
);
    localparam PROD_W = 2 * (DATA_W + 1);

    wire signed [DATA_W:0]       a_ext = {1'b0, a};
    wire signed [DATA_W:0]       b_ext = {b[DATA_W-1], b};
    wire signed [PROD_W-1:0]     product = a_ext * b_ext;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc <= {ACC_W{1'b0}};
        else if (clear)
            acc <= {ACC_W{1'b0}};
        else if (en)
            acc <= acc + {{(ACC_W-PROD_W){product[PROD_W-1]}}, product};
    end
endmodule

module MacArray8x8 #(
    parameter DATA_W = 8,
    parameter ACC_W  = 32
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         clear,
    input  wire                         en,

    input  wire        [8*DATA_W-1:0]   a_vec,
    input  wire signed [8*DATA_W-1:0]   b_vec,
    output wire signed [64*ACC_W-1:0]   acc_mat
);
    genvar r, c;
    generate
        for (r = 0; r < 8; r = r + 1) begin : ROW
            wire [DATA_W-1:0] a_lane = a_vec[r*DATA_W +: DATA_W];
            for (c = 0; c < 8; c = c + 1) begin : COL
                wire signed [DATA_W-1:0] b_lane = b_vec[c*DATA_W +: DATA_W];
                wire signed [ACC_W-1:0] acc_cell;

                MacCell #(
                    .DATA_W(DATA_W),
                    .ACC_W(ACC_W)
                ) u_mac_cell (
                    .clk(clk),
                    .rst_n(rst_n),
                    .clear(clear),
                    .en(en),
                    .a(a_lane),
                    .b(b_lane),
                    .acc(acc_cell)
                );

                assign acc_mat[(r*8 + c)*ACC_W +: ACC_W] = acc_cell;
            end
        end
    endgenerate
endmodule
