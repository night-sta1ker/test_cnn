`timescale 1ns / 1ps

module im2col_tb;
    localparam DATA_W = 8;
    localparam IN_H = 5;
    localparam IN_W = 13;
    localparam OUT_W = IN_W - 2;

    reg clk;
    reg rst_n;
    reg clear;
    reg line_wr_en;
    reg line_wr_buf;
    reg [1:0] line_wr_row;
    reg [2:0] line_wr_word;
    reg [4*DATA_W-1:0] line_wr_data;
    reg start;
    reg start_buf;
    reg [15:0] in_w;
    wire busy;
    wire done;
    wire [DATA_W-1:0] out_data;
    wire out_valid;
    reg out_ready;

    integer errors;

    Im2Col3x3 #(.DATA_W(DATA_W), .LINE_W(32)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear),
        .line_wr_en(line_wr_en),
        .line_wr_buf(line_wr_buf),
        .line_wr_row(line_wr_row),
        .line_wr_word(line_wr_word),
        .line_wr_data(line_wr_data),
        .start(start),
        .start_buf(start_buf),
        .in_w(in_w),
        .busy(busy),
        .done(done),
        .out_data(out_data),
        .out_valid(out_valid),
        .out_ready(out_ready)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    function [DATA_W-1:0] pixel;
        input integer row;
        input integer col;
        begin
            if ((row >= 0) && (row < IN_H) && (col >= 0) && (col < IN_W))
                pixel = row * IN_W + col + 1;
            else
                pixel = 0;
        end
    endfunction

    function [4*DATA_W-1:0] pack_word;
        input integer row;
        input integer word_col;
        integer col;
        begin
            col = word_col * 4;
            pack_word = {pixel(row, col + 3),
                         pixel(row, col + 2),
                         pixel(row, col + 1),
                         pixel(row, col + 0)};
        end
    endfunction

    task write_line_word;
        input integer buf_id;
        input integer local_row;
        input integer image_row;
        input integer word_col;
        begin
            @(negedge clk);
            line_wr_en = 1'b1;
            line_wr_buf = buf_id[0];
            line_wr_row = local_row[1:0];
            line_wr_word = word_col[2:0];
            line_wr_data = pack_word(image_row, word_col);
            @(negedge clk);
            line_wr_en = 1'b0;
        end
    endtask

    task load_tile;
        input integer buf_id;
        input integer base_row;
        integer row;
        integer word_col;
        begin
            for (row = 0; row < 3; row = row + 1) begin
                for (word_col = 0; word_col < 4; word_col = word_col + 1)
                    write_line_word(buf_id, row, base_row + row, word_col);
            end
        end
    endtask

    task run_tile;
        input integer buf_id;
        input integer base_row;
        integer col;
        integer k;
        integer kh;
        integer kw;
        reg [DATA_W-1:0] exp;
        begin
            @(negedge clk);
            start_buf = buf_id[0];
            in_w = IN_W[15:0];
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            for (col = 0; col < OUT_W; col = col + 1) begin
                for (k = 0; k < 9; k = k + 1) begin
                    kh = k / 3;
                    kw = k % 3;
                    exp = pixel(base_row + kh, col + kw);
                    @(negedge clk);
                    if (!out_valid || out_data !== exp) begin
                        $display("IM2COL FAIL: base_row=%0d col=%0d k=%0d data=%0d expected=%0d valid=%b",
                                 base_row, col, k, out_data, exp, out_valid);
                        errors = errors + 1;
                    end
                    out_ready = 1'b1;
                    @(negedge clk);
                    out_ready = 1'b0;
                end
            end

            if (!done) begin
                $display("IM2COL FAIL: done not asserted after tile base_row=%0d", base_row);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        clear = 1'b0;
        line_wr_en = 1'b0;
        line_wr_buf = 1'b0;
        line_wr_row = 2'b0;
        line_wr_word = 3'b0;
        line_wr_data = 32'b0;
        start = 1'b0;
        start_buf = 1'b0;
        in_w = IN_W[15:0];
        out_ready = 1'b0;
        errors = 0;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        load_tile(0, 0);
        run_tile(0, 0);
        load_tile(1, 1);
        run_tile(1, 1);
        load_tile(0, 2);
        run_tile(0, 2);

        if (errors == 0)
            $display("IM2COL TEST PASS");
        else
            $display("IM2COL TEST FAIL: errors=%0d", errors);
        $finish;
    end

endmodule
