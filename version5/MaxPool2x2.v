`timescale 1ns / 1ps

// Fixed unsigned 2x2, stride-2, floor max-pool engine.
module MaxPool2x2 #(
    parameter SRAM_ADDR_W = 10
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   clear,
    input  wire                   start,
    input  wire [11:0]            src_base_byte,
    input  wire [11:0]            dst_base_byte,
    input  wire [3:0]             channels,
    input  wire [5:0]             in_h,
    input  wire [5:0]             in_w,
    input  wire [11:0]            src_plane_stride,

    output wire                   busy,
    output reg                    done,
    output wire                   sram_rd_en,
    output wire [SRAM_ADDR_W-1:0] sram_rd_addr,
    input  wire [31:0]            sram_rd_data,
    output wire                   sram_wr_en,
    output wire [SRAM_ADDR_W-1:0] sram_wr_addr,
    output wire [31:0]            sram_wr_data,
    output wire [3:0]             sram_wr_be
);
    localparam S_IDLE  = 3'd0;
    localparam S_READ  = 3'd1;
    localparam S_WAIT  = 3'd2;
    localparam S_WRITE = 3'd3;

    reg [2:0] state;
    reg [3:0] channel;
    reg [5:0] out_y;
    reg [5:0] out_x;
    reg [1:0] sample;
    reg [11:0] plane_base;
    reg [11:0] row_base;
    reg [11:0] window_base;
    reg [11:0] dst_byte_addr;
    reg [7:0] max_value;

    wire [5:0] out_h = in_h >> 1;
    wire [5:0] out_w = in_w >> 1;
    wire [11:0] sample_byte_addr = window_base +
                                   (sample[1] ? {6'b0, in_w} : 12'b0) +
                                   (sample[0] ? 12'd1 : 12'd0);
    wire [7:0] sample_value = sram_rd_data >> (sample_byte_addr[1:0] * 8);
    wire [7:0] max_next = (sample == 0 || sample_value > max_value) ?
                          sample_value : max_value;

    assign busy = (state != S_IDLE);
    assign sram_rd_en = (state == S_READ);
    assign sram_rd_addr = sample_byte_addr[SRAM_ADDR_W+1:2];
    assign sram_wr_en = (state == S_WRITE);
    assign sram_wr_addr = dst_byte_addr[SRAM_ADDR_W+1:2];
    assign sram_wr_be = 4'b0001 << dst_byte_addr[1:0];
    assign sram_wr_data = {24'b0, max_value} << (dst_byte_addr[1:0] * 8);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            channel <= 0;
            out_y <= 0;
            out_x <= 0;
            sample <= 0;
            plane_base <= 0;
            row_base <= 0;
            window_base <= 0;
            dst_byte_addr <= 0;
            max_value <= 0;
        end
        else if (clear) begin
            state <= S_IDLE;
            done <= 1'b0;
        end
        else begin
            done <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    channel <= 0;
                    out_y <= 0;
                    out_x <= 0;
                    sample <= 0;
                    plane_base <= src_base_byte;
                    row_base <= src_base_byte;
                    window_base <= src_base_byte;
                    dst_byte_addr <= dst_base_byte;
                    max_value <= 0;
                    state <= S_READ;
                end

                S_READ: state <= S_WAIT;

                S_WAIT: begin
                    max_value <= max_next;
                    if (sample == 3)
                        state <= S_WRITE;
                    else begin
                        sample <= sample + 1'b1;
                        state <= S_READ;
                    end
                end

                S_WRITE: begin
                    dst_byte_addr <= dst_byte_addr + 1'b1;
                    sample <= 0;
                    max_value <= 0;
                    if (out_x + 1'b1 < out_w) begin
                        out_x <= out_x + 1'b1;
                        window_base <= window_base + 2;
                        state <= S_READ;
                    end
                    else if (out_y + 1'b1 < out_h) begin
                        out_x <= 0;
                        out_y <= out_y + 1'b1;
                        row_base <= row_base + ({6'b0, in_w} << 1);
                        window_base <= row_base + ({6'b0, in_w} << 1);
                        state <= S_READ;
                    end
                    else if (channel + 1'b1 < channels) begin
                        channel <= channel + 1'b1;
                        out_x <= 0;
                        out_y <= 0;
                        plane_base <= plane_base + src_plane_stride;
                        row_base <= plane_base + src_plane_stride;
                        window_base <= plane_base + src_plane_stride;
                        state <= S_READ;
                    end
                    else begin
                        done <= 1'b1;
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
