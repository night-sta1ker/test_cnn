`timescale 1ns / 1ps

// 22 fixed output-channel parameter entries for Conv1, Conv2 and FC.
module ParamBuffer (
    input  wire         clk,
    input  wire         wr_en,
    input  wire [4:0]   wr_addr,
    input  wire [64:0]  wr_data,
    input  wire [4:0]   rd_base,
    output reg  [127:0] rd_bias,
    output reg  [63:0]  rd_mult,
    output reg  [31:0]  rd_shift,
    output reg  [31:0]  rd_zout,
    output reg  [3:0]   rd_relu
);
    localparam PARAM_COUNT = 22;
    reg [64:0] mem [0:PARAM_COUNT-1];

    always @(posedge clk) begin
        if (wr_en && wr_addr < PARAM_COUNT)
            mem[wr_addr] <= wr_data;
    end

    integer i;
    reg [64:0] entry;
    always @* begin
        rd_bias = 0;
        rd_mult = 0;
        rd_shift = 0;
        rd_zout = 0;
        rd_relu = 0;
        for (i = 0; i < 4; i = i + 1) begin
            entry = ((rd_base + i) < PARAM_COUNT) ? mem[rd_base + i] : 65'b0;
            rd_bias[i*32 +: 32] = entry[31:0];
            rd_mult[i*16 +: 16] = entry[47:32];
            rd_shift[i*8 +: 8] = entry[55:48];
            rd_zout[i*8 +: 8] = entry[63:56];
            rd_relu[i] = entry[64];
        end
    end
endmodule
