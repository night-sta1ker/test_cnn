`timescale 1ns / 1ps

module tb_ProcessUnitMode0;
    localparam IMG_W = 28;
    localparam IMG_H = 28;
    localparam OUT_W = 26;
    localparam OUT_H = 26;
    localparam N_OC  = 32;
    localparam OC_GROUPS = (N_OC + 7) / 8;

    reg clk;
    reg rst_n;
    reg start;

    reg  [7:0] act;
    reg        act_en;
    wire       act_ready;

    reg [7:0] wgt0, wgt1, wgt2, wgt3;
    reg [7:0] wgt4, wgt5, wgt6, wgt7;
    reg       wgt_en;
    wire      wgt_ready;

    reg signed [15:0] mult0, mult1, mult2, mult3;
    reg signed [15:0] mult4, mult5, mult6, mult7;
    reg signed [7:0]  shift_param0, shift_param1, shift_param2, shift_param3;
    reg signed [7:0]  shift_param4, shift_param5, shift_param6, shift_param7;

    wire [63:0] out_parallel;
    wire        out_en;
    wire        done;

    reg [7:0]        act_mem  [0:IMG_H-1][0:IMG_W-1];
    reg signed [7:0] wgt_mem  [0:N_OC-1][0:8];
    reg signed [15:0] mult_mem [0:N_OC-1];
    reg signed [7:0]  shft_mem [0:N_OC-1];

    reg [7:0] cap_out [0:OUT_H-1][0:OUT_W-1][0:N_OC-1];
    reg       cap_val [0:OUT_H-1][0:OUT_W-1][0:N_OC-1];

    integer total_out_cnt;
    integer mac_cycle_cnt;
    integer bad_mac_tiles;
    reg [2:0] prev_state;

    Accelerator #(
        .IMG_W(IMG_W),
        .IMG_H(IMG_H),
        .OUT_W(OUT_W),
        .OUT_H(OUT_H),
        .N_OC(N_OC)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .act(act),
        .act_en(act_en),
        .act_ready(act_ready),
        .wgt0(wgt0), .wgt1(wgt1), .wgt2(wgt2), .wgt3(wgt3),
        .wgt4(wgt4), .wgt5(wgt5), .wgt6(wgt6), .wgt7(wgt7),
        .wgt_en(wgt_en),
        .wgt_ready(wgt_ready),
        .mult0(mult0), .mult1(mult1), .mult2(mult2), .mult3(mult3),
        .mult4(mult4), .mult5(mult5), .mult6(mult6), .mult7(mult7),
        .shift_param0(shift_param0), .shift_param1(shift_param1),
        .shift_param2(shift_param2), .shift_param3(shift_param3),
        .shift_param4(shift_param4), .shift_param5(shift_param5),
        .shift_param6(shift_param6), .shift_param7(shift_param7),
        .out_parallel(out_parallel),
        .out_en(out_en),
        .done(done)
    );

    always #5 clk = ~clk;

    task clear_capture;
        integer r, c, ch;
        begin
            for (r = 0; r < OUT_H; r = r + 1)
                for (c = 0; c < OUT_W; c = c + 1)
                    for (ch = 0; ch < N_OC; ch = ch + 1) begin
                        cap_out[r][c][ch] = 8'd0;
                        cap_val[r][c][ch] = 1'b0;
                    end
            total_out_cnt = 0;
            mac_cycle_cnt = 0;
            bad_mac_tiles = 0;
            prev_state = 3'd0;
        end
    endtask

    function [7:0] requant_ref;
        input signed [31:0] acc_in;
        input signed [15:0] mult_in;
        input signed [7:0]  shift_in;
        reg signed [47:0] prod;
        reg signed [7:0]  q_sum;
        reg [5:0]         shamt;
        reg signed [47:0] qv;
        begin
            prod = acc_in * mult_in;
            q_sum = 8'sd15 + shift_in;
            shamt = (q_sum > 8'sd63) ? 6'd63 :
                    (q_sum < 8'sd0)  ? 6'd0  : q_sum[5:0];
            qv = prod >>> shamt;
            if (qv < 0)
                requant_ref = 8'd0;
            else if (qv > 255)
                requant_ref = 8'd255;
            else
                requant_ref = qv[7:0];
        end
    endfunction

    function [7:0] golden_pixel;
        input [4:0] r;
        input [4:0] c;
        input [4:0] oc;
        reg signed [31:0] acc;
        integer kr, kc;
        begin
            acc = 0;
            for (kr = 0; kr < 3; kr = kr + 1)
                for (kc = 0; kc < 3; kc = kc + 1)
                    acc = acc +
                          $signed({1'b0, act_mem[r + kr][c + kc]}) *
                          $signed(wgt_mem[oc][kr * 3 + kc]);
            golden_pixel = requant_ref(acc, mult_mem[oc], shft_mem[oc]);
        end
    endfunction

    task drive_inputs;
        integer act_idx;
        integer wgt_group_idx;
        integer wgt_k_idx;
        integer oc_base;
        begin
            act_idx = 0;
            wgt_group_idx = 0;
            wgt_k_idx = 0;
            while ((act_idx < IMG_W * IMG_H) || (wgt_group_idx < OC_GROUPS)) begin
                @(negedge clk);
                if (act_ready && (act_idx < IMG_W * IMG_H)) begin
                    act_en <= 1'b1;
                    act    <= act_mem[act_idx / IMG_W][act_idx % IMG_W];
                end else begin
                    act_en <= 1'b0;
                    act    <= 8'd0;
                end

                if (wgt_ready && (wgt_group_idx < OC_GROUPS)) begin
                    oc_base = wgt_group_idx * 8;
                    wgt_en      <= 1'b1;
                    wgt0        <= wgt_mem[oc_base + 0][wgt_k_idx];
                    wgt1        <= wgt_mem[oc_base + 1][wgt_k_idx];
                    wgt2        <= wgt_mem[oc_base + 2][wgt_k_idx];
                    wgt3        <= wgt_mem[oc_base + 3][wgt_k_idx];
                    wgt4        <= wgt_mem[oc_base + 4][wgt_k_idx];
                    wgt5        <= wgt_mem[oc_base + 5][wgt_k_idx];
                    wgt6        <= wgt_mem[oc_base + 6][wgt_k_idx];
                    wgt7        <= wgt_mem[oc_base + 7][wgt_k_idx];
                    mult0       <= mult_mem[oc_base + 0];
                    mult1       <= mult_mem[oc_base + 1];
                    mult2       <= mult_mem[oc_base + 2];
                    mult3       <= mult_mem[oc_base + 3];
                    mult4       <= mult_mem[oc_base + 4];
                    mult5       <= mult_mem[oc_base + 5];
                    mult6       <= mult_mem[oc_base + 6];
                    mult7       <= mult_mem[oc_base + 7];
                    shift_param0 <= shft_mem[oc_base + 0];
                    shift_param1 <= shft_mem[oc_base + 1];
                    shift_param2 <= shft_mem[oc_base + 2];
                    shift_param3 <= shft_mem[oc_base + 3];
                    shift_param4 <= shft_mem[oc_base + 4];
                    shift_param5 <= shft_mem[oc_base + 5];
                    shift_param6 <= shft_mem[oc_base + 6];
                    shift_param7 <= shft_mem[oc_base + 7];
                end else begin
                    wgt_en      <= 1'b0;
                    wgt0 <= 8'd0; wgt1 <= 8'd0; wgt2 <= 8'd0; wgt3 <= 8'd0;
                    wgt4 <= 8'd0; wgt5 <= 8'd0; wgt6 <= 8'd0; wgt7 <= 8'd0;
                    mult0 <= 16'sd0; mult1 <= 16'sd0; mult2 <= 16'sd0; mult3 <= 16'sd0;
                    mult4 <= 16'sd0; mult5 <= 16'sd0; mult6 <= 16'sd0; mult7 <= 16'sd0;
                    shift_param0 <= 8'sd0; shift_param1 <= 8'sd0;
                    shift_param2 <= 8'sd0; shift_param3 <= 8'sd0;
                    shift_param4 <= 8'sd0; shift_param5 <= 8'sd0;
                    shift_param6 <= 8'sd0; shift_param7 <= 8'sd0;
                end

                @(posedge clk);
                if (act_en && act_ready)
                    act_idx = act_idx + 1;
                if (wgt_en && wgt_ready) begin
                    if (wgt_k_idx == 8) begin
                        wgt_k_idx = 0;
                        wgt_group_idx = wgt_group_idx + 1;
                    end else begin
                        wgt_k_idx = wgt_k_idx + 1;
                    end
                end
            end

            @(negedge clk);
            act_en <= 1'b0;
            wgt_en <= 1'b0;
        end
    endtask

    task check_result;
        input [31:0] test_id;
        integer r, c, ch;
        integer err_cnt;
        integer val_cnt;
        reg [7:0] expected;
        begin
            err_cnt = 0;
            val_cnt = 0;

            for (r = 0; r < OUT_H; r = r + 1)
                for (c = 0; c < OUT_W; c = c + 1)
                    for (ch = 0; ch < N_OC; ch = ch + 1) begin
                        if (cap_val[r][c][ch])
                            val_cnt = val_cnt + 1;
                        expected = golden_pixel(r[4:0], c[4:0], ch[4:0]);
                        if (!cap_val[r][c][ch] || (cap_out[r][c][ch] !== expected)) begin
                            if (err_cnt < 12)
                                $display("MISMATCH test=%0d r=%0d c=%0d ch=%0d valid=%0b got=%0d exp=%0d",
                                         test_id, r, c, ch, cap_val[r][c][ch],
                                         cap_out[r][c][ch], expected);
                            err_cnt = err_cnt + 1;
                        end
                    end

            if ((err_cnt == 0) && (val_cnt == OUT_H * OUT_W * N_OC) && (bad_mac_tiles == 0))
                $display("PASS test=%0d valid=%0d out_en_pulses=%0d", test_id, val_cnt, total_out_cnt);
            else begin
                $display("FAIL test=%0d errors=%0d valid=%0d bad_mac_tiles=%0d",
                         test_id, err_cnt, val_cnt, bad_mac_tiles);
                $stop;
            end
        end
    endtask

    task run_one_test;
        input [31:0] test_id;
        begin
            clear_capture();

            rst_n = 1'b0;
            repeat (5) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);

            @(negedge clk);
            start <= 1'b1;
            @(negedge clk);
            start <= 1'b0;

            drive_inputs();
            wait(done);
            repeat (5) @(posedge clk);
            check_result(test_id);
        end
    endtask

    task gen_test_impulse_kernel;
        integer r, c, ch, k;
        begin
            for (r = 0; r < IMG_H; r = r + 1)
                for (c = 0; c < IMG_W; c = c + 1)
                    act_mem[r][c] = (r * IMG_W + c) & 8'hff;

            for (ch = 0; ch < N_OC; ch = ch + 1) begin
                for (k = 0; k < 9; k = k + 1)
                    wgt_mem[ch][k] = 8'sd0;
                wgt_mem[ch][0] = 8'sd1;
                wgt_mem[ch][8] = 8'sd2;
                mult_mem[ch] = 16'sd32767;
                shft_mem[ch] = 8'sd0;
            end
        end
    endtask

    task gen_test_mixed;
        integer r, c, ch, k;
        begin
            for (r = 0; r < IMG_H; r = r + 1)
                for (c = 0; c < IMG_W; c = c + 1)
                    act_mem[r][c] = (r * 7 + c * 11 + 3) & 8'hff;

            for (ch = 0; ch < N_OC; ch = ch + 1) begin
                for (k = 0; k < 9; k = k + 1)
                    wgt_mem[ch][k] = ((ch + k) % 3 == 0) ? 8'sd2 :
                                      ((ch + k) % 3 == 1) ? -8'sd1 : 8'sd1;
                mult_mem[ch] = 16'sd16384;
                shft_mem[ch] = 8'sd0;
            end
        end
    endtask

    reg [4:0] cap_row;
    reg [4:0] cap_col_base;
    reg [2:0] cap_lane;
    reg [1:0] cap_oc_group;
    integer   cap_col;
    integer   cap_oc_base;

    always @(posedge clk) begin
        if (!rst_n) begin
            mac_cycle_cnt <= 0;
            prev_state <= uut.process_unit_u.state;
        end else begin
            if (uut.process_unit_u.state == uut.process_unit_u.S_MAC)
                mac_cycle_cnt <= mac_cycle_cnt + 1;
            if ((prev_state == uut.process_unit_u.S_MAC) && (uut.process_unit_u.state == uut.process_unit_u.S_STEP)) begin
                if (mac_cycle_cnt != 9) begin
                    $display("BAD MAC TIMING row=%0d col_base=%0d oc_group=%0d cycles=%0d",
                             uut.process_unit_u.out_row, uut.process_unit_u.out_col_base, uut.process_unit_u.oc_group, mac_cycle_cnt);
                    bad_mac_tiles <= bad_mac_tiles + 1;
                end
                mac_cycle_cnt <= 0;
            end
            prev_state <= uut.process_unit_u.state;
        end

        #1;
        if (out_en) begin
            cap_row = uut.out_dbg_row;
            cap_col_base = uut.out_dbg_col_base;
            cap_lane = uut.out_dbg_lane;
            cap_oc_group = uut.out_dbg_oc_group;
            cap_col = cap_col_base + cap_lane;
            cap_oc_base = cap_oc_group * 8;
            cap_out[cap_row][cap_col][cap_oc_base + 0] = out_parallel[7:0];
            cap_out[cap_row][cap_col][cap_oc_base + 1] = out_parallel[15:8];
            cap_out[cap_row][cap_col][cap_oc_base + 2] = out_parallel[23:16];
            cap_out[cap_row][cap_col][cap_oc_base + 3] = out_parallel[31:24];
            cap_out[cap_row][cap_col][cap_oc_base + 4] = out_parallel[39:32];
            cap_out[cap_row][cap_col][cap_oc_base + 5] = out_parallel[47:40];
            cap_out[cap_row][cap_col][cap_oc_base + 6] = out_parallel[55:48];
            cap_out[cap_row][cap_col][cap_oc_base + 7] = out_parallel[63:56];
            cap_val[cap_row][cap_col][cap_oc_base + 0] = 1'b1;
            cap_val[cap_row][cap_col][cap_oc_base + 1] = 1'b1;
            cap_val[cap_row][cap_col][cap_oc_base + 2] = 1'b1;
            cap_val[cap_row][cap_col][cap_oc_base + 3] = 1'b1;
            cap_val[cap_row][cap_col][cap_oc_base + 4] = 1'b1;
            cap_val[cap_row][cap_col][cap_oc_base + 5] = 1'b1;
            cap_val[cap_row][cap_col][cap_oc_base + 6] = 1'b1;
            cap_val[cap_row][cap_col][cap_oc_base + 7] = 1'b1;
            total_out_cnt = total_out_cnt + 8;
        end
    end

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        act = 8'd0;
        act_en = 1'b0;
        wgt0 = 8'd0; wgt1 = 8'd0; wgt2 = 8'd0; wgt3 = 8'd0;
        wgt4 = 8'd0; wgt5 = 8'd0; wgt6 = 8'd0; wgt7 = 8'd0;
        wgt_en = 1'b0;
        mult0 = 16'sd0; mult1 = 16'sd0; mult2 = 16'sd0; mult3 = 16'sd0;
        mult4 = 16'sd0; mult5 = 16'sd0; mult6 = 16'sd0; mult7 = 16'sd0;
        shift_param0 = 8'sd0; shift_param1 = 8'sd0;
        shift_param2 = 8'sd0; shift_param3 = 8'sd0;
        shift_param4 = 8'sd0; shift_param5 = 8'sd0;
        shift_param6 = 8'sd0; shift_param7 = 8'sd0;

        $display("Test 1: impulse edge kernel");
        gen_test_impulse_kernel();
        run_one_test(1);

        $display("Test 2: mixed signed kernel");
        gen_test_mixed();
        run_one_test(2);

        $display("All ProcessUnitMode0 tests passed.");
        $finish;
    end

    initial begin
        #10000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule
