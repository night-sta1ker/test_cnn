`timescale 1ns / 1ps

`ifndef E2E_FIRST_INDEX
`define E2E_FIRST_INDEX 0
`endif
`ifndef E2E_CASES
`define E2E_CASES 10
`endif

module npu_inference_core_tb;
    localparam E2E_FIRST_INDEX = `E2E_FIRST_INDEX;
    localparam E2E_CASES = `E2E_CASES;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg start = 1'b0;
    wire busy;
    wire done;
    wire [3:0] class_index;
    wire [7:0] class_score;
    wire error;

    reg host_act_wr_en = 1'b0;
    reg [9:0] host_act_wr_addr = 0;
    reg [31:0] host_act_wr_data = 0;
    reg [3:0] host_act_wr_be = 0;
    reg model_wgt_wr_en = 1'b0;
    reg [9:0] model_wgt_wr_addr = 0;
    reg [31:0] model_wgt_wr_data = 0;
    reg model_param_wr_en = 1'b0;
    reg [4:0] model_param_wr_addr = 0;
    reg [64:0] model_param_wr_data = 0;

    reg [31:0] input_words [0:E2E_CASES*196-1];
    reg [31:0] weight_words [0:680];
    reg [64:0] param_words [0:21];
    reg [7:0] golden_conv1 [0:E2E_CASES*2704-1];
    reg [7:0] golden_pool1 [0:E2E_CASES*676-1];
    reg [7:0] golden_conv2 [0:E2E_CASES*968-1];
    reg [7:0] golden_pool2 [0:E2E_CASES*200-1];
    reg [7:0] golden_fc [0:E2E_CASES*10-1];
    reg [7:0] golden_label [0:E2E_CASES-1];

    integer errors = 0;
    integer cycles = 0;
    integer conv1_cycles = 0;
    integer conv1_array_cycles = 0;
    integer conv2_cycles = 0;
    integer conv2_array_cycles = 0;
    integer fc_cycles = 0;
    integer fc_array_cycles = 0;
    integer current_case = 0;
    reg error_seen = 1'b0;

    always #5 clk = ~clk;

    NpuInferenceCore dut (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
        .class_index(class_index), .class_score(class_score), .error(error),
        .host_act_wr_en(host_act_wr_en), .host_act_wr_addr(host_act_wr_addr),
        .host_act_wr_data(host_act_wr_data), .host_act_wr_be(host_act_wr_be),
        .model_wgt_wr_en(model_wgt_wr_en),
        .model_wgt_wr_addr(model_wgt_wr_addr),
        .model_wgt_wr_data(model_wgt_wr_data),
        .model_param_wr_en(model_param_wr_en),
        .model_param_wr_addr(model_param_wr_addr),
        .model_param_wr_data(model_param_wr_data)
    );

    function [7:0] sram_byte;
        input integer byte_addr;
        reg [31:0] word;
        begin
            word = dut.activation_sram_u.mem[byte_addr >> 2];
            case (byte_addr & 3)
                0: sram_byte = word[7:0];
                1: sram_byte = word[15:8];
                2: sram_byte = word[23:16];
                default: sram_byte = word[31:24];
            endcase
        end
    endfunction

    function [7:0] golden_byte;
        input integer region;
        input integer index;
        begin
            case (region)
                1: golden_byte = golden_conv1[current_case * 2704 + index];
                2: golden_byte = golden_pool1[current_case * 676 + index];
                3: golden_byte = golden_conv2[current_case * 968 + index];
                4: golden_byte = golden_pool2[current_case * 200 + index];
                default: golden_byte = golden_fc[current_case * 10 + index];
            endcase
        end
    endfunction

    task check_region;
        input integer region;
        input integer base_byte;
        input integer count;
        integer i;
        integer local_errors;
        reg [7:0] actual;
        reg [7:0] expected;
        begin
            #1;
            local_errors = 0;
            for (i = 0; i < count; i = i + 1) begin
                actual = sram_byte(base_byte + i);
                expected = golden_byte(region, i);
                if (actual !== expected) begin
                    if (local_errors < 12)
                        $display("E2E MISMATCH region=%0d index=%0d expected=%02x actual=%02x",
                                 region, i, expected, actual);
                    local_errors = local_errors + 1;
                end
            end
            errors = errors + local_errors;
            if (local_errors != 0)
                $display("E2E REGION FAIL region=%0d errors=%0d",
                         region, local_errors);
        end
    endtask

    integer i;
    integer expected_class;
    integer case_error_base;
    integer case_start_cycle;
    integer passed_cases;
    reg [7:0] expected_score;
    initial begin
        $readmemh("e2e_input.hex", input_words);
        $readmemh("model_weights.hex", weight_words);
        $readmemh("model_params_hw.hex", param_words);
        $readmemh("e2e_conv1.hex", golden_conv1);
        $readmemh("e2e_pool1.hex", golden_pool1);
        $readmemh("e2e_conv2.hex", golden_conv2);
        $readmemh("e2e_pool2.hex", golden_pool2);
        $readmemh("e2e_fc.hex", golden_fc);
        $readmemh("e2e_label.hex", golden_label);

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // All preload ports must be blocked once an inference is active.
        model_wgt_wr_en = 1'b1;
        model_wgt_wr_addr = 0;
        model_wgt_wr_data = 32'h11223344;
        model_param_wr_en = 1'b1;
        model_param_wr_addr = 0;
        model_param_wr_data = 65'h1_01234567_89abcdef;
        host_act_wr_en = 1'b1;
        host_act_wr_addr = 0;
        host_act_wr_data = 32'h55667788;
        host_act_wr_be = 4'hf;
        @(negedge clk);
        model_wgt_wr_en = 1'b0;
        model_param_wr_en = 1'b0;
        host_act_wr_en = 1'b0;
        host_act_wr_be = 0;

        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        model_wgt_wr_en = 1'b1;
        model_wgt_wr_data = 32'hdeadbeef;
        model_param_wr_en = 1'b1;
        model_param_wr_data = 65'h0_deadbeef_cafefeed;
        host_act_wr_en = 1'b1;
        host_act_wr_data = 32'hcafebabe;
        host_act_wr_be = 4'hf;
        @(negedge clk);
        model_wgt_wr_en = 1'b0;
        model_param_wr_en = 1'b0;
        host_act_wr_en = 1'b0;
        host_act_wr_be = 0;
        #1;
        if (!error || dut.wgt_buffer_u.mem[0] !== 32'h11223344 ||
            dut.param_buffer_u.mem[0] !== 65'h1_01234567_89abcdef ||
            dut.activation_sram_u.mem[0] !== 32'h55667788)
            $fatal(1, "TOP write protection or sticky error failed");

        rst_n = 1'b0;
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        error_seen = 1'b0;
        cycles = 0;

        for (i = 0; i < 681; i = i + 1) begin
            model_wgt_wr_en = 1'b1;
            model_wgt_wr_addr = i;
            model_wgt_wr_data = weight_words[i];
            @(negedge clk);
        end
        model_wgt_wr_en = 1'b0;

        for (i = 0; i < 22; i = i + 1) begin
            model_param_wr_en = 1'b1;
            model_param_wr_addr = i;
            model_param_wr_data = param_words[i];
            @(negedge clk);
        end
        model_param_wr_en = 1'b0;

        passed_cases = 0;
        for (current_case = 0; current_case < E2E_CASES;
             current_case = current_case + 1) begin
            case_error_base = errors;
            error_seen = 1'b0;

            @(negedge clk);
            for (i = 0; i < 196; i = i + 1) begin
                host_act_wr_en = 1'b1;
                host_act_wr_addr = i;
                host_act_wr_data = input_words[current_case * 196 + i];
                host_act_wr_be = 4'hf;
                @(negedge clk);
            end
            host_act_wr_en = 1'b0;
            host_act_wr_be = 0;
            case_start_cycle = cycles;

            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            wait (dut.stage == 3'd2);
            check_region(1, 1024, 2704);
            wait (dut.stage == 3'd3);
            check_region(2, 0, 676);
            wait (dut.stage == 3'd4);
            check_region(3, 1024, 968);
            wait (dut.stage == 3'd5);
            check_region(4, 0, 200);
            wait (dut.stage == 3'd6);
            check_region(5, 1024, 10);
            wait (done);
            #1;

            expected_class = 0;
            expected_score = golden_fc[current_case * 10];
            for (i = 1; i < 10; i = i + 1)
                if (golden_fc[current_case * 10 + i] > expected_score) begin
                    expected_class = i;
                    expected_score = golden_fc[current_case * 10 + i];
                end

            if (class_index !== expected_class || class_score !== expected_score) begin
                $display("E2E CLASS FAIL index=%0d expected=%0d/%0d actual=%0d/%0d",
                         E2E_FIRST_INDEX + current_case, expected_class,
                         expected_score, class_index, class_score);
                errors = errors + 1;
            end
            if (class_index !== golden_label[current_case][3:0]) begin
                $display("E2E LABEL FAIL index=%0d label=%0d actual=%0d",
                         E2E_FIRST_INDEX + current_case,
                         golden_label[current_case], class_index);
                errors = errors + 1;
            end
            if (error_seen) begin
                $display("E2E ERROR SIGNAL ASSERTED index=%0d",
                         E2E_FIRST_INDEX + current_case);
                errors = errors + 1;
            end

            if (errors == case_error_base) begin
                passed_cases = passed_cases + 1;
                $display("E2E CASE PASS index=%0d label=%0d class=%0d score=%0d cycles=%0d",
                         E2E_FIRST_INDEX + current_case,
                         golden_label[current_case], class_index, class_score,
                         cycles - case_start_cycle);
            end else begin
                $display("E2E CASE FAIL index=%0d new_errors=%0d",
                         E2E_FIRST_INDEX + current_case,
                         errors - case_error_base);
            end
        end

        if (errors == 0)
            $display("NPU E2E TEST PASS cases=%0d/%0d total_cycles=%0d",
                     passed_cases, E2E_CASES, cycles);
        else
            $display("NPU E2E TEST FAIL cases=%0d/%0d errors=%0d",
                     passed_cases, E2E_CASES, errors);
        $display("ARRAY UTIL conv1 stage_cycles=%0d enabled_cycles=%0d",
                 conv1_cycles, conv1_array_cycles);
        $display("ARRAY UTIL conv2 stage_cycles=%0d enabled_cycles=%0d",
                 conv2_cycles, conv2_array_cycles);
        $display("ARRAY UTIL fc stage_cycles=%0d enabled_cycles=%0d",
                 fc_cycles, fc_array_cycles);
        $finish;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            cycles <= 0;
            conv1_cycles <= 0;
            conv1_array_cycles <= 0;
            conv2_cycles <= 0;
            conv2_array_cycles <= 0;
            fc_cycles <= 0;
            fc_array_cycles <= 0;
            error_seen <= 1'b0;
        end else begin
            cycles <= cycles + 1;
            if (dut.stage == 3'd1) begin
                conv1_cycles <= conv1_cycles + 1;
                if (dut.array_en)
                    conv1_array_cycles <= conv1_array_cycles + 1;
            end
            if (dut.stage == 3'd3) begin
                conv2_cycles <= conv2_cycles + 1;
                if (dut.array_en)
                    conv2_array_cycles <= conv2_array_cycles + 1;
            end
            if (dut.stage == 3'd5) begin
                fc_cycles <= fc_cycles + 1;
                if (dut.array_en)
                    fc_array_cycles <= fc_array_cycles + 1;
            end
            if (error)
                error_seen <= 1'b1;
            if (cycles > 500000) begin
                $display("NPU E2E TEST TIMEOUT stage=%0d", dut.stage);
                $finish;
            end
        end
    end
endmodule
