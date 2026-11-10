`timescale 1ns / 1ps

module wgt_buffer_tb;
    localparam ADDR_W = 10;
    localparam MODEL_WORDS = 681;

    reg clk;
    reg rst_n;
    reg clear;
    reg wr_en;
    reg [ADDR_W-1:0] wr_addr;
    reg [31:0] wr_data;
    reg base_valid;
    reg [ADDR_W-1:0] base_addr;
    reg [15:0] k_size;
    wire [31:0] stream_data;
    wire stream_valid;
    reg stream_ready;

    integer errors;
    integer i;
    integer seen;
    integer cycle;
    reg [31:0] stalled_data;

    WgtBuffer dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear),
        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .base_valid(base_valid),
        .base_addr(base_addr),
        .k_size(k_size),
        .stream_data(stream_data),
        .stream_valid(stream_valid),
        .stream_ready(stream_ready)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    function [31:0] weight_word;
        input integer addr;
        begin
            weight_word = 32'ha5000000 ^ (addr * 32'h00010101);
        end
    endfunction

    task write_weight;
        input integer addr;
        begin
            @(negedge clk);
            wr_en = 1'b1;
            wr_addr = addr[ADDR_W-1:0];
            wr_data = weight_word(addr);
            @(negedge clk);
            wr_en = 1'b0;
        end
    endtask

    task request_stream;
        input integer addr;
        input integer words;
        begin
            @(negedge clk);
            base_addr = addr[ADDR_W-1:0];
            k_size = words[15:0];
            base_valid = 1'b1;
            @(negedge clk);
            base_valid = 1'b0;
        end
    endtask

    task check_stream;
        input integer addr;
        input integer words;
        input integer stall_pattern;
        integer timeout;
        begin
            seen = 0;
            cycle = 0;
            timeout = 0;
            stream_ready = 1'b0;
            while (seen < words && timeout < 10000) begin
                @(negedge clk);
                case (stall_pattern)
                    0: stream_ready = 1'b1;
                    1: stream_ready = !cycle[0];
                    default: stream_ready = (cycle % 11) < 4;
                endcase

                if (stream_valid && !stream_ready) begin
                    stalled_data = stream_data;
                    @(posedge clk);
                    #1;
                    if (stream_valid && stream_data !== stalled_data) begin
                        $display("WGT FAIL data changed under backpressure old=%h new=%h",
                                 stalled_data, stream_data);
                        errors = errors + 1;
                    end
                end
                else if (stream_valid && stream_ready) begin
                    if (stream_data !== weight_word(addr + seen)) begin
                        $display("WGT FAIL base=%0d index=%0d data=%h expected=%h",
                                 addr, seen, stream_data, weight_word(addr + seen));
                        errors = errors + 1;
                    end
                    seen = seen + 1;
                end

                cycle = cycle + 1;
                timeout = timeout + 1;
            end
            if (seen != words) begin
                $display("WGT FAIL timeout base=%0d seen=%0d expected=%0d", addr, seen, words);
                errors = errors + 1;
            end
            // Keep ready asserted through the edge that consumes the final word.
            @(posedge clk);
            #1;
            stream_ready = 1'b0;
            @(negedge clk);
            if (stream_valid) begin
                $display("WGT FAIL stream_valid stayed high after base=%0d", addr);
                errors = errors + 1;
            end
        end
    endtask

    task run_stream;
        input integer addr;
        input integer words;
        input integer stall_pattern;
        begin
            request_stream(addr, words);
            check_stream(addr, words, stall_pattern);
        end
    endtask

    initial begin
        rst_n = 1'b0;
        clear = 1'b0;
        wr_en = 1'b0;
        wr_addr = {ADDR_W{1'b0}};
        wr_data = 32'b0;
        base_valid = 1'b0;
        base_addr = {ADDR_W{1'b0}};
        k_size = 16'b0;
        stream_ready = 1'b0;
        stalled_data = 32'b0;
        errors = 0;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        for (i = 0; i < MODEL_WORDS; i = i + 1)
            write_weight(i);

        run_stream(0, 9, 0);      // Conv1
        run_stream(9, 36, 1);     // Conv2 tile 0
        run_stream(9, 36, 2);     // Conv2 M-block replay
        run_stream(45, 36, 0);    // Conv2 tile 1
        run_stream(81, 200, 1);   // FC tile 0
        run_stream(281, 200, 2);  // FC tile 1
        run_stream(481, 200, 0);  // FC tile 2, ending at word 680

        // A zero-length request produces no stream.
        request_stream(0, 0);
        repeat (3) @(negedge clk);
        if (stream_valid) begin
            $display("WGT FAIL zero-length request asserted valid");
            errors = errors + 1;
        end

        // Clear aborts an active replay but preserves resident weights.
        request_stream(9, 36);
        wait (stream_valid);
        @(negedge clk);
        clear = 1'b1;
        @(negedge clk);
        clear = 1'b0;
        if (stream_valid) begin
            $display("WGT FAIL clear did not drop stream_valid");
            errors = errors + 1;
        end
        run_stream(9, 36, 0);

        if (errors == 0) begin
            $display("WGT BUFFER TEST PASS");
            $finish;
        end
        else begin
            $display("WGT BUFFER TEST FAIL errors=%0d", errors);
            $fatal(1, "WGT BUFFER TEST FAILED");
        end
    end

endmodule
