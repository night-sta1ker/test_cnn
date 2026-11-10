`timescale 1ns / 1ps

module act_buffer_tb;
    localparam CONV_WORDS = 26 * 7 * 9;
    localparam CONV2_WORDS = 11 * 3 * 4 * 9;

    reg clk;
    reg rst_n;
    reg clear;
    reg cfg_valid;
    wire cfg_ready;
    reg cfg_mode;
    reg [15:0] cfg_h;
    reg [15:0] cfg_w;
    reg [2:0] cfg_channels;
    reg [15:0] cfg_channel_stride;
    reg [15:0] cfg_vector_size;
    reg req_valid;
    reg [11:0] req_base_byte;
    wire sram_rd_en;
    wire [9:0] sram_rd_addr;
    wire [31:0] sram_rd_data;
    wire [31:0] seq_data;
    wire seq_valid;
    reg seq_ready;

    reg sram_wr_en;
    reg [9:0] sram_wr_addr;
    reg [31:0] sram_wr_data;
    reg [3:0] sram_wr_be;

    integer errors;
    integer seen;
    integer cycle;
    integer timeout;
    integer i;
    reg stalled;
    reg [31:0] stalled_data;

    ActivationSram sram_u (
        .clk(clk),
        .rd_en(sram_rd_en), .rd_addr(sram_rd_addr), .rd_data(sram_rd_data),
        .wr_en(sram_wr_en), .wr_addr(sram_wr_addr),
        .wr_data(sram_wr_data), .wr_be(sram_wr_be)
    );

    ActBuffer dut (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready), .cfg_mode(cfg_mode),
        .cfg_h(cfg_h), .cfg_w(cfg_w), .cfg_channels(cfg_channels),
        .cfg_channel_stride(cfg_channel_stride), .cfg_vector_size(cfg_vector_size),
        .req_valid(req_valid), .req_base_byte(req_base_byte),
        .sram_rd_en(sram_rd_en), .sram_rd_addr(sram_rd_addr),
        .sram_rd_data(sram_rd_data),
        .seq_data(seq_data), .seq_valid(seq_valid), .seq_ready(seq_ready)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    function [7:0] pixel;
        input integer index;
        begin
            pixel = (index * 13 + 8'h35) & 8'hff;
        end
    endfunction

    function [7:0] signed_pixel;
        input integer row;
        input integer col;
        begin
            signed_pixel = pixel(row * 28 + col) ^ 8'h80;
        end
    endfunction

    function [31:0] input_word;
        input integer word_index;
        integer byte_index;
        begin
            byte_index = word_index * 4;
            input_word = {pixel(byte_index + 3), pixel(byte_index + 2),
                          pixel(byte_index + 1), pixel(byte_index)};
        end
    endfunction

    function [31:0] conv_expected;
        input integer word_index;
        integer y;
        integer rem;
        integer group;
        integer kh;
        integer kw;
        integer x;
        reg [7:0] lane0;
        reg [7:0] lane1;
        reg [7:0] lane2;
        reg [7:0] lane3;
        begin
            y = word_index / (7 * 9);
            rem = word_index % (7 * 9);
            group = rem / 9;
            kh = (rem % 9) / 3;
            kw = rem % 3;
            x = group * 4;
            lane0 = (x < 26) ? signed_pixel(y + kh, x + kw) : 0;
            lane1 = (x + 1 < 26) ? signed_pixel(y + kh, x + kw + 1) : 0;
            lane2 = (x + 2 < 26) ? signed_pixel(y + kh, x + kw + 2) : 0;
            lane3 = (x + 3 < 26) ? signed_pixel(y + kh, x + kw + 3) : 0;
            conv_expected = {lane3, lane2, lane1, lane0};
        end
    endfunction

    function [31:0] conv2_expected;
        input integer word_index;
        integer y;
        integer rem;
        integer group;
        integer ch;
        integer kh;
        integer kw;
        integer x;
        reg [7:0] lane0;
        reg [7:0] lane1;
        reg [7:0] lane2;
        reg [7:0] lane3;
        begin
            y = word_index / (3 * 4 * 9);
            rem = word_index % (3 * 4 * 9);
            group = rem / (4 * 9);
            rem = rem % (4 * 9);
            ch = rem / 9;
            kh = (rem % 9) / 3;
            kw = rem % 3;
            x = group * 4;
            lane0 = (x < 11) ? (pixel(ch * 169 + (y + kh) * 13 + x + kw) ^ 8'h80) : 0;
            lane1 = (x + 1 < 11) ? (pixel(ch * 169 + (y + kh) * 13 + x + kw + 1) ^ 8'h80) : 0;
            lane2 = (x + 2 < 11) ? (pixel(ch * 169 + (y + kh) * 13 + x + kw + 2) ^ 8'h80) : 0;
            lane3 = (x + 3 < 11) ? (pixel(ch * 169 + (y + kh) * 13 + x + kw + 3) ^ 8'h80) : 0;
            conv2_expected = {lane3, lane2, lane1, lane0};
        end
    endfunction

    task write_sram_word;
        input integer addr;
        begin
            @(negedge clk);
            sram_wr_en = 1'b1;
            sram_wr_addr = addr[9:0];
            sram_wr_data = input_word(addr);
            sram_wr_be = 4'hf;
            @(negedge clk);
            sram_wr_en = 1'b0;
        end
    endtask

    task configure;
        input mode;
        input integer height;
        input integer width;
        input integer channels;
        input integer channel_stride;
        input integer vector_size;
        begin
            wait (cfg_ready);
            @(negedge clk);
            cfg_mode = mode;
            cfg_h = height[15:0];
            cfg_w = width[15:0];
            cfg_channels = channels[2:0];
            cfg_channel_stride = channel_stride[15:0];
            cfg_vector_size = vector_size[15:0];
            cfg_valid = 1'b1;
            @(negedge clk);
            cfg_valid = 1'b0;
        end
    endtask

    task request;
        begin
            @(negedge clk);
            req_base_byte = 12'd0;
            req_valid = 1'b1;
            @(negedge clk);
            req_valid = 1'b0;
        end
    endtask

    task check_stream;
        input integer words;
        input integer pattern;
        input integer is_fc;
        reg [31:0] expected;
        begin
            seen = 0;
            cycle = 0;
            timeout = 0;
            stalled = 1'b0;
            while ((seen < words) && (timeout < 200000)) begin
                @(negedge clk);
                case (pattern)
                    0: seq_ready = 1'b1;
                    1: seq_ready = (cycle % 5) != 0;
                    default: seq_ready = (cycle % 17) < 3;
                endcase
                if (seq_valid) begin
                    if (!seq_ready) begin
                        if (!stalled) begin
                            stalled = 1'b1;
                            stalled_data = seq_data;
                        end
                        else if (seq_data !== stalled_data) begin
                            $display("ACT FAIL data changed under backpressure");
                            errors = errors + 1;
                        end
                    end
                    else begin
                        expected = (is_fc == 1) ? {24'b0, (pixel(seen) ^ 8'h80)} :
                                   (is_fc == 2) ? conv2_expected(seen) : conv_expected(seen);
                        if (seq_data !== expected) begin
                            $display("ACT FAIL index=%0d data=%h expected=%h", seen, seq_data, expected);
                            errors = errors + 1;
                        end
                        seen = seen + 1;
                        stalled = 1'b0;
                    end
                end
                cycle = cycle + 1;
                timeout = timeout + 1;
            end
            if (seen != words) begin
                $display("ACT FAIL timeout seen=%0d expected=%0d", seen, words);
                errors = errors + 1;
            end
            @(posedge clk);
            #1;
            seq_ready = 1'b0;
            @(negedge clk);
            if (seq_valid) begin
                $display("ACT FAIL seq_valid remained high after final transfer");
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        clear = 1'b0;
        cfg_valid = 1'b0;
        cfg_mode = 1'b0;
        cfg_h = 0;
        cfg_w = 0;
        cfg_channels = 0;
        cfg_channel_stride = 0;
        cfg_vector_size = 0;
        req_valid = 1'b0;
        req_base_byte = 0;
        seq_ready = 1'b0;
        sram_wr_en = 1'b0;
        sram_wr_addr = 0;
        sram_wr_data = 0;
        sram_wr_be = 0;
        errors = 0;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        for (i = 0; i < 196; i = i + 1)
            write_sram_word(i);

        configure(1'b0, 28, 28, 1, 784, 0);
        request;
        check_stream(CONV_WORDS, 1, 0);

        // Sequencer requests the same activation again for another N tile.
        request;
        check_stream(CONV_WORDS, 2, 0);

        // Conv2 uses unaligned 13-byte rows and 169-byte CHW channel strides.
        for (i = 0; i < 169; i = i + 1)
            write_sram_word(i);
        configure(1'b0, 13, 13, 4, 169, 0);
        request;
        check_stream(CONV2_WORDS, 2, 2);

        configure(1'b1, 0, 0, 0, 0, 200);
        request;
        check_stream(200, 1, 1);

        if (errors == 0) begin
            $display("ACT BUFFER TEST PASS");
            $finish;
        end
        else begin
            $display("ACT BUFFER TEST FAIL errors=%0d", errors);
            $fatal(1, "ACT BUFFER TEST FAILED");
        end
    end
endmodule
