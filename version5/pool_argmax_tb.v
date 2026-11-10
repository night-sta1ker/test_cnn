`timescale 1ns / 1ps

module pool_argmax_tb;
    reg clk;
    reg rst_n;
    reg clear;

    reg pool_start;
    reg [11:0] pool_src_base;
    reg [11:0] pool_dst_base;
    reg [3:0] pool_channels;
    reg [5:0] pool_h;
    reg [5:0] pool_w;
    reg [11:0] pool_plane_stride;
    wire pool_busy;
    wire pool_done;
    wire pool_rd_en;
    wire [9:0] pool_rd_addr;
    wire pool_wr_en;
    wire [9:0] pool_wr_addr;
    wire [31:0] pool_wr_data;
    wire [3:0] pool_wr_be;

    reg arg_start;
    reg [11:0] arg_src_base;
    wire arg_busy;
    wire arg_done;
    wire [3:0] class_index;
    wire [7:0] arg_max;
    wire arg_rd_en;
    wire [9:0] arg_rd_addr;

    reg tb_wr_en;
    reg [9:0] tb_wr_addr;
    reg [31:0] tb_wr_data;
    reg [3:0] tb_wr_be;
    wire sram_rd_en = pool_rd_en | arg_rd_en;
    wire [9:0] sram_rd_addr = pool_rd_en ? pool_rd_addr : arg_rd_addr;
    wire [31:0] sram_rd_data;
    wire sram_wr_en = tb_wr_en | pool_wr_en;
    wire [9:0] sram_wr_addr = tb_wr_en ? tb_wr_addr : pool_wr_addr;
    wire [31:0] sram_wr_data = tb_wr_en ? tb_wr_data : pool_wr_data;
    wire [3:0] sram_wr_be = tb_wr_en ? tb_wr_be : pool_wr_be;

    integer errors;
    integer c;
    integer y;
    integer x;
    integer addr;
    integer expected;

    ActivationSram sram_u (
        .clk(clk),
        .rd_en(sram_rd_en), .rd_addr(sram_rd_addr), .rd_data(sram_rd_data),
        .wr_en(sram_wr_en), .wr_addr(sram_wr_addr),
        .wr_data(sram_wr_data), .wr_be(sram_wr_be)
    );

    MaxPool2x2 pool_u (
        .clk(clk), .rst_n(rst_n), .clear(clear), .start(pool_start),
        .src_base_byte(pool_src_base), .dst_base_byte(pool_dst_base),
        .channels(pool_channels), .in_h(pool_h), .in_w(pool_w),
        .src_plane_stride(pool_plane_stride),
        .busy(pool_busy), .done(pool_done),
        .sram_rd_en(pool_rd_en), .sram_rd_addr(pool_rd_addr),
        .sram_rd_data(sram_rd_data),
        .sram_wr_en(pool_wr_en), .sram_wr_addr(pool_wr_addr),
        .sram_wr_data(pool_wr_data), .sram_wr_be(pool_wr_be)
    );

    ArgMax10 arg_u (
        .clk(clk), .rst_n(rst_n), .clear(clear), .start(arg_start),
        .src_base_byte(arg_src_base), .busy(arg_busy), .done(arg_done),
        .class_index(class_index), .max_value(arg_max),
        .sram_rd_en(arg_rd_en), .sram_rd_addr(arg_rd_addr),
        .sram_rd_data(sram_rd_data)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    function [7:0] source_value;
        input integer kind;
        input integer channel;
        input integer row;
        input integer col;
        begin
            source_value = (kind * 53 + channel * 37 + row * 11 + col * 7) & 8'hff;
        end
    endfunction

    function [7:0] stored_byte;
        input integer byte_addr;
        begin
            stored_byte = sram_u.mem[byte_addr >> 2] >> ((byte_addr & 3) * 8);
        end
    endfunction

    task write_byte;
        input integer byte_addr;
        input [7:0] value;
        begin
            @(negedge clk);
            tb_wr_en = 1'b1;
            tb_wr_addr = byte_addr >> 2;
            tb_wr_be = 4'b0001 << (byte_addr & 3);
            tb_wr_data = {24'b0, value} << ((byte_addr & 3) * 8);
            @(negedge clk);
            tb_wr_en = 1'b0;
        end
    endtask

    task load_pool_source;
        input integer kind;
        input integer channels;
        input integer height;
        input integer width;
        begin
            for (c = 0; c < channels; c = c + 1)
                for (y = 0; y < height; y = y + 1)
                    for (x = 0; x < width; x = x + 1)
                        write_byte(1024 + c * height * width + y * width + x,
                                   source_value(kind, c, y, x));
        end
    endtask

    task run_pool;
        input integer kind;
        input integer channels;
        input integer height;
        input integer width;
        integer out_h;
        integer out_w;
        integer v0;
        integer v1;
        integer v2;
        integer v3;
        begin
            out_h = height / 2;
            out_w = width / 2;
            pool_src_base = 12'd1024;
            pool_dst_base = 12'd0;
            pool_channels = channels[3:0];
            pool_h = height[5:0];
            pool_w = width[5:0];
            pool_plane_stride = (height * width);
            @(negedge clk);
            pool_start = 1'b1;
            @(negedge clk);
            pool_start = 1'b0;
            wait (pool_done);
            @(negedge clk);

            for (c = 0; c < channels; c = c + 1) begin
                for (y = 0; y < out_h; y = y + 1) begin
                    for (x = 0; x < out_w; x = x + 1) begin
                        v0 = source_value(kind, c, y * 2, x * 2);
                        v1 = source_value(kind, c, y * 2, x * 2 + 1);
                        v2 = source_value(kind, c, y * 2 + 1, x * 2);
                        v3 = source_value(kind, c, y * 2 + 1, x * 2 + 1);
                        expected = (v0 > v1) ? v0 : v1;
                        if (v2 > expected) expected = v2;
                        if (v3 > expected) expected = v3;
                        addr = c * out_h * out_w + y * out_w + x;
                        if (stored_byte(addr) !== expected[7:0]) begin
                            $display("POOL FAIL kind=%0d c=%0d y=%0d x=%0d data=%0d expected=%0d",
                                     kind, c, y, x, stored_byte(addr), expected);
                            errors = errors + 1;
                        end
                    end
                end
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        clear = 1'b0;
        pool_start = 1'b0;
        pool_src_base = 0;
        pool_dst_base = 0;
        pool_channels = 0;
        pool_h = 0;
        pool_w = 0;
        pool_plane_stride = 0;
        arg_start = 1'b0;
        arg_src_base = 0;
        tb_wr_en = 1'b0;
        tb_wr_addr = 0;
        tb_wr_data = 0;
        tb_wr_be = 0;
        errors = 0;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        load_pool_source(1, 4, 26, 26);
        run_pool(1, 4, 26, 26);

        load_pool_source(2, 8, 11, 11);
        run_pool(2, 8, 11, 11);

        write_byte(1024, 8'd3);
        write_byte(1025, 8'd200);
        write_byte(1026, 8'd55);
        write_byte(1027, 8'd200);
        write_byte(1028, 8'd7);
        write_byte(1029, 8'd99);
        write_byte(1030, 8'd1);
        write_byte(1031, 8'd42);
        write_byte(1032, 8'd88);
        write_byte(1033, 8'd17);
        arg_src_base = 12'd1024;
        @(negedge clk);
        arg_start = 1'b1;
        @(negedge clk);
        arg_start = 1'b0;
        wait (arg_done);
        @(negedge clk);
        if (class_index != 1 || arg_max != 200) begin
            $display("ARGMAX FAIL class=%0d max=%0d expected_class=1 expected_max=200",
                     class_index, arg_max);
            errors = errors + 1;
        end

        if (errors == 0) begin
            $display("POOL ARGMAX TEST PASS");
            $finish;
        end
        else begin
            $display("POOL ARGMAX TEST FAIL errors=%0d", errors);
            $fatal(1, "POOL ARGMAX TEST FAILED");
        end
    end
endmodule
