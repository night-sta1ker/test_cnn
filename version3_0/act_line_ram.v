`timescale 1ns / 1ps

module dpram #(
    parameter DATA_W = 8,
    parameter ADDR_W = 8,
    parameter DEPTH  = 256
) (
    input  wire                 clk,

    input  wire [ADDR_W-1:0]    addra,
    input  wire                 ena,
    input  wire                 wea,
    input  wire [DATA_W-1:0]    dina,

    input  wire [ADDR_W-1:0]    addrb,
    input  wire                 enb,
    output reg  [DATA_W-1:0]    doutb
);

    reg [DATA_W-1:0] BRAM [0:DEPTH-1];

    always @(posedge clk) begin
        if (ena && wea)
            BRAM[addra] <= dina;

        if (enb)
            doutb <= BRAM[addrb];
    end

endmodule

module act_line_ram #(
    parameter DATA_W = 8,
    parameter MAX_K_SIZE = 128,
    parameter BANK_DEPTH = (MAX_K_SIZE + 3) / 4,
    parameter BANK_ADDR_W = 8
) (
    input  wire                     clk,

    input  wire [3:0]               wr_en,
    input  wire [4*16-1:0]          wr_k,
    input  wire [4*DATA_W-1:0]      wr_data,

    input  wire                     rd_en,
    input  wire [15:0]              rd_k,
    output wire signed [DATA_W-1:0] rd_data
);

    wire [1:0] rd_bank = rd_k[1:0];

    reg [BANK_ADDR_W-1:0] wr_addr [0:3];
    reg                   wr_bank_en [0:3];
    reg [DATA_W-1:0]      wr_bank_data [0:3];

    wire [BANK_ADDR_W-1:0] rd_addr = rd_k[15:2];
    wire [DATA_W-1:0]      bank_dout [0:3];
    reg [1:0]              rd_bank_d;

    integer lane;
    reg [15:0] wr_k_lane;
    reg [1:0] wr_bank_lane;

    always @(*) begin
        for (lane = 0; lane < 4; lane = lane + 1) begin
            wr_addr[lane] = {BANK_ADDR_W{1'b0}};
            wr_bank_en[lane] = 1'b0;
            wr_bank_data[lane] = {DATA_W{1'b0}};
        end

        for (lane = 0; lane < 4; lane = lane + 1) begin
            wr_k_lane = wr_k[lane*16 +: 16];
            wr_bank_lane = wr_k_lane[1:0];
            if (wr_en[lane]) begin
                wr_addr[wr_bank_lane] = wr_k_lane[15:2];
                wr_bank_en[wr_bank_lane] = 1'b1;
                wr_bank_data[wr_bank_lane] = wr_data[lane*DATA_W +: DATA_W];
            end
        end
    end

    always @(posedge clk) begin
        if (rd_en)
            rd_bank_d <= rd_bank;
    end

    dpram #(
        .DATA_W(DATA_W),
        .ADDR_W(BANK_ADDR_W),
        .DEPTH(BANK_DEPTH)
    ) bank0_u (
        .clk(clk),
        .addra(wr_addr[0]),
        .ena(wr_bank_en[0]),
        .wea(wr_bank_en[0]),
        .dina(wr_bank_data[0]),
        .addrb(rd_addr),
        .enb(rd_en && (rd_bank == 2'd0)),
        .doutb(bank_dout[0])
    );

    dpram #(
        .DATA_W(DATA_W),
        .ADDR_W(BANK_ADDR_W),
        .DEPTH(BANK_DEPTH)
    ) bank1_u (
        .clk(clk),
        .addra(wr_addr[1]),
        .ena(wr_bank_en[1]),
        .wea(wr_bank_en[1]),
        .dina(wr_bank_data[1]),
        .addrb(rd_addr),
        .enb(rd_en && (rd_bank == 2'd1)),
        .doutb(bank_dout[1])
    );

    dpram #(
        .DATA_W(DATA_W),
        .ADDR_W(BANK_ADDR_W),
        .DEPTH(BANK_DEPTH)
    ) bank2_u (
        .clk(clk),
        .addra(wr_addr[2]),
        .ena(wr_bank_en[2]),
        .wea(wr_bank_en[2]),
        .dina(wr_bank_data[2]),
        .addrb(rd_addr),
        .enb(rd_en && (rd_bank == 2'd2)),
        .doutb(bank_dout[2])
    );

    dpram #(
        .DATA_W(DATA_W),
        .ADDR_W(BANK_ADDR_W),
        .DEPTH(BANK_DEPTH)
    ) bank3_u (
        .clk(clk),
        .addra(wr_addr[3]),
        .ena(wr_bank_en[3]),
        .wea(wr_bank_en[3]),
        .dina(wr_bank_data[3]),
        .addrb(rd_addr),
        .enb(rd_en && (rd_bank == 2'd3)),
        .doutb(bank_dout[3])
    );

    assign rd_data = bank_dout[rd_bank_d];

endmodule
