`timescale 1ns / 1ps

// 3x3 stride-1 im2col generator for one output row.
//
// Two 3-row line-buffer banks are provided so a caller can ping-pong between
// loading the next 3-row window and consuming the current one.
module Im2Col3x3 #(
    parameter DATA_W = 8,
    parameter LINE_W = 32,
    parameter WORDS_PER_ROW = LINE_W / 4
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clear,

    // 32-bit row-aligned line-buffer write.
    input  wire                    line_wr_en,
    input  wire                    line_wr_buf,
    input  wire [1:0]              line_wr_row,
    input  wire [2:0]              line_wr_word,
    input  wire [4*DATA_W-1:0]     line_wr_data,

    // Generate im2col values for one output row from one buffer bank.
    input  wire                    start,
    input  wire                    start_buf,
    input  wire [15:0]             in_w,
    output reg                     busy,
    output reg                     done,

    output wire [DATA_W-1:0]       out_data,
    output wire                    out_valid,
    input  wire                    out_ready
);

    reg [4*DATA_W-1:0] buf0 [0:2][0:WORDS_PER_ROW-1];
    reg [4*DATA_W-1:0] buf1 [0:2][0:WORDS_PER_ROW-1];

    reg        active_buf;
    reg [15:0] out_w;
    reg [15:0] out_col;
    reg [3:0]  k_cnt;

    wire [1:0] kh = (k_cnt < 4'd3) ? 2'd0 : (k_cnt < 4'd6) ? 2'd1 : 2'd2;
    wire [1:0] kw = (k_cnt == 4'd0 || k_cnt == 4'd3 || k_cnt == 4'd6) ? 2'd0 :
                    (k_cnt == 4'd1 || k_cnt == 4'd4 || k_cnt == 4'd7) ? 2'd1 : 2'd2;
    wire [15:0] rd_col = out_col + kw;
    wire [2:0]  rd_word_idx = rd_col[4:2];
    wire [1:0]  rd_byte_idx = rd_col[1:0];
    wire [4*DATA_W-1:0] rd_word = active_buf ? buf1[kh][rd_word_idx] : buf0[kh][rd_word_idx];

    assign out_valid = busy;
    assign out_data  = rd_word[rd_byte_idx*DATA_W +: DATA_W];

    always @(posedge clk) begin
        if (line_wr_en) begin
            if (line_wr_buf)
                buf1[line_wr_row][line_wr_word] <= line_wr_data;
            else
                buf0[line_wr_row][line_wr_word] <= line_wr_data;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_buf <= 1'b0;
            out_w      <= 16'b0;
            out_col    <= 16'b0;
            k_cnt      <= 4'b0;
            busy       <= 1'b0;
            done       <= 1'b0;
        end
        else if (clear) begin
            active_buf <= 1'b0;
            out_w      <= 16'b0;
            out_col    <= 16'b0;
            k_cnt      <= 4'b0;
            busy       <= 1'b0;
            done       <= 1'b0;
        end
        else begin
            done <= 1'b0;

            if (start && !busy) begin
                active_buf <= start_buf;
                out_w      <= in_w - 16'd2;
                out_col    <= 16'b0;
                k_cnt      <= 4'b0;
                busy       <= (in_w >= 16'd3);
                done       <= (in_w < 16'd3);
            end
            else if (out_valid && out_ready) begin
                if ((k_cnt == 4'd8) && (out_col == out_w - 16'd1)) begin
                    busy    <= 1'b0;
                    done    <= 1'b1;
                    k_cnt   <= 4'b0;
                    out_col <= 16'b0;
                end
                else if (k_cnt == 4'd8) begin
                    k_cnt   <= 4'b0;
                    out_col <= out_col + 1'b1;
                end
                else begin
                    k_cnt <= k_cnt + 1'b1;
                end
            end
        end
    end

endmodule
