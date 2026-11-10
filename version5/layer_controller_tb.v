`timescale 1ns / 1ps

module layer_controller_tb;
    reg clk;
    reg rst_n;
    reg start;
    reg gemm_done;
    reg pool_done;
    reg argmax_done;
    wire busy;
    wire done;
    wire gemm_start;
    wire pool_start;
    wire argmax_start;
    wire [2:0] stage;
    wire act_mode;
    wire [15:0] act_h;
    wire [15:0] act_w;
    wire [2:0] act_channels;
    wire [15:0] act_channel_stride;
    wire [15:0] act_vector_size;
    wire [11:0] act_base_byte;
    wire [15:0] seq_m;
    wire [15:0] seq_k;
    wire [15:0] seq_n;
    wire [9:0] wgt_base_word;
    wire [11:0] result_base_byte;
    wire [5:0] result_out_w;
    wire [5:0] result_row_stride;
    wire [11:0] result_plane_size;
    wire [4:0] param_base;
    wire [11:0] pool_src_base_byte;
    wire [11:0] pool_dst_base_byte;
    wire [3:0] pool_channels;
    wire [5:0] pool_h;
    wire [5:0] pool_w;
    wire [11:0] pool_plane_stride;
    wire [11:0] argmax_src_base_byte;
    integer errors;

    LayerController dut (.*);

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task pulse_gemm_done;
        begin
            @(negedge clk);
            gemm_done = 1'b1;
            @(negedge clk);
            gemm_done = 1'b0;
        end
    endtask

    task pulse_pool_done;
        begin
            @(negedge clk);
            pool_done = 1'b1;
            @(negedge clk);
            pool_done = 1'b0;
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        gemm_done = 1'b0;
        pool_done = 1'b0;
        argmax_done = 1'b0;
        errors = 0;
        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        if (!gemm_start || stage != 1 || seq_m != 728 || seq_k != 9 ||
            seq_n != 4 || act_h != 28 || act_w != 28 ||
            result_plane_size != 676 || param_base != 0) begin
            $display("CTRL FAIL CONV1 configuration");
            errors = errors + 1;
        end

        // A second start while busy must not restart the schedule.
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        if (stage != 1 || gemm_start) begin
            $display("CTRL FAIL accepted start while busy");
            errors = errors + 1;
        end

        pulse_gemm_done;
        if (!pool_start || stage != 2 || pool_channels != 4 ||
            pool_h != 26 || pool_w != 26 || pool_plane_stride != 676) begin
            $display("CTRL FAIL POOL1 configuration");
            errors = errors + 1;
        end

        pulse_pool_done;
        if (!gemm_start || stage != 3 || seq_m != 132 || seq_k != 36 ||
            seq_n != 8 || act_channels != 4 || act_channel_stride != 169 ||
            wgt_base_word != 9 || result_plane_size != 121 || param_base != 4) begin
            $display("CTRL FAIL CONV2 configuration");
            errors = errors + 1;
        end

        pulse_gemm_done;
        if (!pool_start || stage != 4 || pool_channels != 8 ||
            pool_h != 11 || pool_w != 11 || pool_plane_stride != 121) begin
            $display("CTRL FAIL POOL2 configuration");
            errors = errors + 1;
        end

        pulse_pool_done;
        if (!gemm_start || stage != 5 || !act_mode || act_vector_size != 200 ||
            seq_m != 1 || seq_k != 200 || seq_n != 10 ||
            wgt_base_word != 81 || param_base != 12) begin
            $display("CTRL FAIL FC configuration");
            errors = errors + 1;
        end

        pulse_gemm_done;
        if (!argmax_start || stage != 6 || argmax_src_base_byte != 1024) begin
            $display("CTRL FAIL ARGMAX configuration");
            errors = errors + 1;
        end

        @(negedge clk);
        argmax_done = 1'b1;
        @(negedge clk);
        argmax_done = 1'b0;
        if (!done || busy || stage != 0) begin
            $display("CTRL FAIL completion done=%b busy=%b stage=%0d", done, busy, stage);
            errors = errors + 1;
        end

        if (errors == 0) begin
            $display("LAYER CONTROLLER TEST PASS");
            $finish;
        end
        else begin
            $display("LAYER CONTROLLER TEST FAIL errors=%0d", errors);
            $fatal(1, "LAYER CONTROLLER TEST FAILED");
        end
    end
endmodule
