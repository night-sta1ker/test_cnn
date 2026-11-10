`timescale 1ns / 1ps

module ArgMax10 #(
    parameter SRAM_ADDR_W = 10
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   clear,
    input  wire                   start,
    input  wire [11:0]            src_base_byte,
    output wire                   busy,
    output reg                    done,
    output reg  [3:0]             class_index,
    output reg  [7:0]             max_value,
    output wire                   sram_rd_en,
    output wire [SRAM_ADDR_W-1:0] sram_rd_addr,
    input  wire [31:0]            sram_rd_data
);
    localparam S_IDLE = 2'd0;
    localparam S_READ = 2'd1;
    localparam S_WAIT = 2'd2;

    reg [1:0] state;
    reg [3:0] index;
    reg [11:0] byte_addr;
    wire [7:0] value = sram_rd_data >> (byte_addr[1:0] * 8);

    assign busy = (state != S_IDLE);
    assign sram_rd_en = (state == S_READ);
    assign sram_rd_addr = byte_addr[SRAM_ADDR_W+1:2];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            index <= 0;
            byte_addr <= 0;
            class_index <= 0;
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
                    index <= 0;
                    byte_addr <= src_base_byte;
                    class_index <= 0;
                    max_value <= 0;
                    state <= S_READ;
                end
                S_READ: state <= S_WAIT;
                S_WAIT: begin
                    if (index == 0 || value > max_value) begin
                        max_value <= value;
                        class_index <= index;
                    end
                    if (index == 9) begin
                        done <= 1'b1;
                        state <= S_IDLE;
                    end
                    else begin
                        index <= index + 1'b1;
                        byte_addr <= byte_addr + 1'b1;
                        state <= S_READ;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
