`timescale 1ns / 1ps

// 4KB activation store: one synchronous read port and one byte-write port.
module ActivationSram #(
    parameter ADDR_W = 10,
    parameter DEPTH  = (1 << ADDR_W)
) (
    input  wire                  clk,

    input  wire                  rd_en,
    input  wire [ADDR_W-1:0]     rd_addr,
    output reg  [31:0]           rd_data,

    input  wire                  wr_en,
    input  wire [ADDR_W-1:0]     wr_addr,
    input  wire [31:0]           wr_data,
    input  wire [3:0]            wr_be
);

    reg [31:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (wr_en) begin
            if (wr_be[0]) mem[wr_addr][7:0]   <= wr_data[7:0];
            if (wr_be[1]) mem[wr_addr][15:8]  <= wr_data[15:8];
            if (wr_be[2]) mem[wr_addr][23:16] <= wr_data[23:16];
            if (wr_be[3]) mem[wr_addr][31:24] <= wr_data[31:24];
        end

        if (rd_en)
            rd_data <= mem[rd_addr];
    end

endmodule
