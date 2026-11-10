`timescale 1ns / 1ps

module activation_sram_tb;
    localparam ADDR_W = 10;
    localparam DEPTH = 1 << ADDR_W;

    reg clk;
    reg rd_en;
    reg [ADDR_W-1:0] rd_addr;
    wire [31:0] rd_data;
    reg wr_en;
    reg [ADDR_W-1:0] wr_addr;
    reg [31:0] wr_data;
    reg [3:0] wr_be;

    integer errors;
    integer i;
    reg [31:0] expected;

    ActivationSram dut (
        .clk(clk),
        .rd_en(rd_en),
        .rd_addr(rd_addr),
        .rd_data(rd_data),
        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .wr_be(wr_be)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    function [31:0] pattern;
        input integer addr;
        begin
            pattern = 32'h5a5a0000 ^ (addr * 32'h00010101);
        end
    endfunction

    task write_word;
        input [ADDR_W-1:0] addr;
        input [31:0] data;
        input [3:0] be;
        begin
            @(negedge clk);
            wr_en = 1'b1;
            wr_addr = addr;
            wr_data = data;
            wr_be = be;
            @(negedge clk);
            wr_en = 1'b0;
            wr_be = 4'b0;
        end
    endtask

    task read_check;
        input [ADDR_W-1:0] addr;
        input [31:0] data;
        begin
            @(negedge clk);
            rd_en = 1'b1;
            rd_addr = addr;
            @(posedge clk);
            #1;
            if (rd_data !== data) begin
                $display("ACT_SRAM FAIL addr=%0d data=%h expected=%h", addr, rd_data, data);
                errors = errors + 1;
            end
            @(negedge clk);
            rd_en = 1'b0;
        end
    endtask

    initial begin
        rd_en = 1'b0;
        rd_addr = {ADDR_W{1'b0}};
        wr_en = 1'b0;
        wr_addr = {ADDR_W{1'b0}};
        wr_data = 32'b0;
        wr_be = 4'b0;
        errors = 0;

        // Check the full 4KB address range.
        for (i = 0; i < DEPTH; i = i + 1)
            write_word(i[ADDR_W-1:0], pattern(i), 4'b1111);
        for (i = 0; i < DEPTH; i = i + 1)
            read_check(i[ADDR_W-1:0], pattern(i));

        // Byte enables are required by packed uint8 activation writes.
        expected = pattern(17);
        expected[7:0] = 8'hef;
        expected[23:16] = 8'had;
        write_word(10'd17, 32'hdeadbeef, 4'b0101);
        read_check(10'd17, expected);

        // Region A ends below word 256; Region B starts at word 256.
        read_check(10'd195, pattern(195));
        read_check(10'd256, pattern(256));
        read_check(10'd931, pattern(931));

        // Independent read and write ports operate in the same cycle.
        @(negedge clk);
        rd_en = 1'b1;
        rd_addr = 10'd100;
        wr_en = 1'b1;
        wr_addr = 10'd900;
        wr_data = 32'h12345678;
        wr_be = 4'b1111;
        @(posedge clk);
        #1;
        if (rd_data !== pattern(100)) begin
            $display("ACT_SRAM FAIL simultaneous read data=%h expected=%h", rd_data, pattern(100));
            errors = errors + 1;
        end
        @(negedge clk);
        rd_en = 1'b0;
        wr_en = 1'b0;
        wr_be = 4'b0;
        read_check(10'd900, 32'h12345678);

        if (errors == 0)
            $display("ACTIVATION SRAM TEST PASS");
        else
            $display("ACTIVATION SRAM TEST FAIL errors=%0d", errors);
        $finish;
    end

endmodule
