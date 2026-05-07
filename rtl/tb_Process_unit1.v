`timescale 1ns / 1ps

module tb_Process_unit;

    reg         clk;
    reg         rst_n;
    reg         start;
    reg  [7:0]  act;
    reg         act_en;
    wire        act_ready;
    reg  [7:0]  wgt_pos0, wgt_pos1, wgt_pos2;
    reg  [7:0]  wgt_pos3, wgt_pos4, wgt_pos5;
    reg  [7:0]  wgt_pos6, wgt_pos7, wgt_pos8;
    reg         wgt_en;
    reg  signed [31:0] mult;
    reg  signed [7:0] shift_param;

    wire [63:0] out_parallel;
    wire        out_en;
    wire        done;

    Process_unit uut (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .act        (act),
        .act_en     (act_en),
        .act_ready  (act_ready),
        .wgt_pos0   (wgt_pos0), .wgt_pos1(wgt_pos1), .wgt_pos2(wgt_pos2),
        .wgt_pos3   (wgt_pos3), .wgt_pos4(wgt_pos4), .wgt_pos5(wgt_pos5),
        .wgt_pos6   (wgt_pos6), .wgt_pos7(wgt_pos7), .wgt_pos8(wgt_pos8),
        .wgt_en     (wgt_en),
        .mult       (mult),
        .shift_param(shift_param),
        .out_parallel(out_parallel),
        .out_en     (out_en),
        .done       (done)
    );

    always #5 clk = ~clk;

    localparam IMG_W = 28;
    localparam IMG_H = 28;
    localparam N_OC  = 16;
    localparam OUT_W = 26;
    localparam OUT_H = 26;

    reg [7:0]         act_mem  [0:IMG_H-1][0:IMG_W-1];
    reg signed [7:0]  wgt_mem  [0:N_OC-1][0:8];
    reg signed [31:0] mult_mem [0:N_OC-1];
    reg signed [7:0] shft_mem [0:N_OC-1];

    reg [7:0] cap_out [0:OUT_H-1][0:OUT_W-1][0:N_OC-1];
    reg       cap_val [0:OUT_H-1][0:OUT_W-1][0:N_OC-1];
    reg [4:0] mon_row, mon_col;
    reg       lo_done;   // 0=expecting lo, 1=expecting hi
    integer   total_out_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mon_row       <= 5'd0;
            mon_col       <= 5'd0;
            lo_done       <= 1'b0;
            total_out_cnt <= 0;
        end else if (out_en) begin
            total_out_cnt <= total_out_cnt + 8;
            if (!lo_done) begin
                // low 8ch
                cap_out[mon_row][mon_col][0] <= out_parallel[7:0];
                cap_out[mon_row][mon_col][1] <= out_parallel[15:8];
                cap_out[mon_row][mon_col][2] <= out_parallel[23:16];
                cap_out[mon_row][mon_col][3] <= out_parallel[31:24];
                cap_out[mon_row][mon_col][4] <= out_parallel[39:32];
                cap_out[mon_row][mon_col][5] <= out_parallel[47:40];
                cap_out[mon_row][mon_col][6] <= out_parallel[55:48];
                cap_out[mon_row][mon_col][7] <= out_parallel[63:56];
                cap_val[mon_row][mon_col][0] <= 1'b1;
                cap_val[mon_row][mon_col][1] <= 1'b1;
                cap_val[mon_row][mon_col][2] <= 1'b1;
                cap_val[mon_row][mon_col][3] <= 1'b1;
                cap_val[mon_row][mon_col][4] <= 1'b1;
                cap_val[mon_row][mon_col][5] <= 1'b1;
                cap_val[mon_row][mon_col][6] <= 1'b1;
                cap_val[mon_row][mon_col][7] <= 1'b1;
                lo_done <= 1'b1;
            end else begin
                // high 8ch
                cap_out[mon_row][mon_col][8]  <= out_parallel[7:0];
                cap_out[mon_row][mon_col][9]  <= out_parallel[15:8];
                cap_out[mon_row][mon_col][10] <= out_parallel[23:16];
                cap_out[mon_row][mon_col][11] <= out_parallel[31:24];
                cap_out[mon_row][mon_col][12] <= out_parallel[39:32];
                cap_out[mon_row][mon_col][13] <= out_parallel[47:40];
                cap_out[mon_row][mon_col][14] <= out_parallel[55:48];
                cap_out[mon_row][mon_col][15] <= out_parallel[63:56];
                cap_val[mon_row][mon_col][8]  <= 1'b1;
                cap_val[mon_row][mon_col][9]  <= 1'b1;
                cap_val[mon_row][mon_col][10] <= 1'b1;
                cap_val[mon_row][mon_col][11] <= 1'b1;
                cap_val[mon_row][mon_col][12] <= 1'b1;
                cap_val[mon_row][mon_col][13] <= 1'b1;
                cap_val[mon_row][mon_col][14] <= 1'b1;
                cap_val[mon_row][mon_col][15] <= 1'b1;
                lo_done <= 1'b0;
                if (mon_col == OUT_W - 1) begin
                    mon_col <= 5'd0;
                    mon_row <= mon_row + 5'd1;
                end else begin
                    mon_col <= mon_col + 5'd1;
                end
            end
        end
    end

    function [7:0] golden_pixel;
        input [4:0] r, c, oc;
        reg signed [31:0] g_acc;
        reg signed [63:0] g_tmp;
        reg signed [7:0]  g_tmp2;
        reg [5:0] g_shamt;
        reg signed [31:0] g_val;
        integer ki, kj;
        begin
            g_acc = 0;
            for (ki = 0; ki < 3; ki = ki + 1) begin
                for (kj = 0; kj < 3; kj = kj + 1) begin
                    g_acc = g_acc +
                        $signed({1'b0, act_mem[r+ki][c+kj]}) *
                        $signed(wgt_mem[oc][ki*3 + kj]);
                end
            end
            g_tmp   = g_acc * mult_mem[oc];
            g_tmp2 = $signed({1'b0, 6'd31}) + $signed(shft_mem[oc]);
            g_shamt = (g_tmp2 > 8'sd63) ? 6'd63 : (g_tmp2 < 8'sd0) ? 6'd0 : g_tmp2[5:0];
            g_val   = $signed(g_tmp >>> g_shamt);
            if ($signed(g_val) < $signed(32'd0))
                golden_pixel = 8'd0;
            else if ($signed(g_val) > $signed(32'd255))
                golden_pixel = 8'd255;
            else
                golden_pixel = g_val[7:0];
        end
    endfunction

    task clear_capture;
        integer rr, cc, hh;
        begin
            for (rr = 0; rr < OUT_H; rr = rr + 1)
                for (cc = 0; cc < OUT_W; cc = cc + 1)
                    for (hh = 0; hh < N_OC; hh = hh + 1) begin
                        cap_out[rr][cc][hh] = 8'd0;
                        cap_val[rr][cc][hh] = 1'b0;
                    end
        end
    endtask

    task load_init;
        integer col_idx;
        begin
            for (col_idx = 0; col_idx < 56; col_idx = col_idx + 1) begin
                @(posedge clk);
                if (col_idx < 16) begin
                    wgt_en      <= 1'b1;
                    wgt_pos0    <= wgt_mem[col_idx][0];
                    wgt_pos1    <= wgt_mem[col_idx][1];
                    wgt_pos2    <= wgt_mem[col_idx][2];
                    wgt_pos3    <= wgt_mem[col_idx][3];
                    wgt_pos4    <= wgt_mem[col_idx][4];
                    wgt_pos5    <= wgt_mem[col_idx][5];
                    wgt_pos6    <= wgt_mem[col_idx][6];
                    wgt_pos7    <= wgt_mem[col_idx][7];
                    wgt_pos8    <= wgt_mem[col_idx][8];
                    mult        <= mult_mem[col_idx];
                    shift_param <= shft_mem[col_idx];
                end else begin
                    wgt_en      <= 1'b0;
                    wgt_pos0    <= 8'd0; wgt_pos1 <= 8'd0; wgt_pos2 <= 8'd0;
                    wgt_pos3    <= 8'd0; wgt_pos4 <= 8'd0; wgt_pos5 <= 8'd0;
                    wgt_pos6    <= 8'd0; wgt_pos7 <= 8'd0; wgt_pos8 <= 8'd0;
                    mult        <= 32'd0;
                    shift_param <= 8'sd0;
                end
                act_en <= 1'b1;
                if (col_idx < 28)
                    act <= act_mem[0][col_idx];
                else
                    act <= act_mem[1][col_idx - 28];
            end
            @(posedge clk);
            act_en <= 1'b0;
            wgt_en <= 1'b0;
        end
    endtask

    task drive_act;
        input [4:0] mem_row;
        input [4:0] mem_col;
        begin
            while (!act_ready) @(posedge clk);
            act_en <= 1'b1;
            act    <= act_mem[mem_row][mem_col];
            @(posedge clk);
            act_en <= 1'b0;
        end
    endtask

    task run_one_test;
        input [31:0] test_id;
        integer r, c, ch, err_cnt, val_cnt;
        integer in_row, in_col;
        reg [7:0] expected;
        begin
            clear_capture();

            rst_n = 1'b0;
            repeat(5) @(posedge clk);
            rst_n = 1'b1;
            repeat(2) @(posedge clk);

            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;

            load_init();

            for (in_row = 2; in_row < IMG_H; in_row = in_row + 1) begin
                for (in_col = 0; in_col < IMG_W; in_col = in_col + 1)
                    drive_act(in_row[4:0], in_col[4:0]);
            end

            wait(done);
            repeat(5) @(posedge clk);

            err_cnt = 0;
            val_cnt = 0;
            for (r = 0; r < OUT_H; r = r + 1) begin
                for (c = 0; c < OUT_W; c = c + 1) begin
                    for (ch = 0; ch < N_OC; ch = ch + 1) begin
                        if (cap_val[r][c][ch]) begin
                            val_cnt = val_cnt + 1;
                            expected = golden_pixel(r[4:0], c[4:0], ch[4:0]);
                            if (cap_out[r][c][ch] !== expected) begin
                                if (err_cnt < 10)
                                    $display("  MISMATCH [r=%0d c=%0d ch=%0d]: got=%0d exp=%0d",
                                             r, c, ch, cap_out[r][c][ch], expected);
                                err_cnt = err_cnt + 1;
                            end
                        end
                    end
                end
            end

            if (err_cnt == 0 && val_cnt == OUT_H * OUT_W * N_OC)
                $display("  PASS: %0d / %0d", val_cnt, OUT_H * OUT_W * N_OC);
            else
                $display("  FAIL: %0d mismatches / %0d valid", err_cnt, val_cnt);
        end
    endtask

    // Test data generators
    task gen_test_ramp;
        integer i, j, ch;
        begin
            for (i = 0; i < IMG_H; i = i + 1)
                for (j = 0; j < IMG_W; j = j + 1)
                    act_mem[i][j] = (i * IMG_W + j) % 256;
            for (ch = 0; ch < N_OC; ch = ch + 1) begin
                for (i = 0; i < 9; i = i + 1)
                    wgt_mem[ch][i] = ((ch + i) % 2 == 0) ? 8'd1 : -8'd1;
                mult_mem[ch]  = 32'd128;
                shft_mem[ch]  = -8'sd24;
            end
        end
    endtask

    task gen_test_all255;
        integer i, j, ch;
        begin
            for (i = 0; i < IMG_H; i = i + 1)
                for (j = 0; j < IMG_W; j = j + 1)
                    act_mem[i][j] = 8'd255;
            for (ch = 0; ch < N_OC; ch = ch + 1) begin
                for (i = 0; i < 9; i = i + 1)
                    wgt_mem[ch][i] = ((ch + i) % 2 == 0) ? 8'd127 : -8'd127;
                mult_mem[ch]  = 32'd256;
                shft_mem[ch]  = -8'sd20;
            end
        end
    endtask

    task gen_test_random;
        integer i, j, ch, w;
        begin
            for (i = 0; i < IMG_H; i = i + 1)
                for (j = 0; j < IMG_W; j = j + 1)
                    act_mem[i][j] = {$random} % 256;
            for (ch = 0; ch < N_OC; ch = ch + 1) begin
                for (w = 0; w < 9; w = w + 1)
                    wgt_mem[ch][w] = {$random} % 255 - 127;
                mult_mem[ch]  = {$random} % 2048 + 1;
                shft_mem[ch]  = {$random} % 25 - 20;
            end
        end
    endtask

    initial begin
        clk         = 1'b0;
        rst_n       = 1'b0;
        start       = 1'b0;
        act         = 8'd0;
        act_en      = 1'b0;
        wgt_pos0    = 8'd0; wgt_pos1 = 8'd0; wgt_pos2 = 8'd0;
        wgt_pos3    = 8'd0; wgt_pos4 = 8'd0; wgt_pos5 = 8'd0;
        wgt_pos6    = 8'd0; wgt_pos7 = 8'd0; wgt_pos8 = 8'd0;
        wgt_en      = 1'b0;
        mult        = 32'd0;
        shift_param = 8'sd0;

        $display("============================================");
        $display("  Test 1: Ramp");
        $display("============================================");
        gen_test_ramp();
        run_one_test(1);

        $display("============================================");
        $display("  Test 2: All-255");
        $display("============================================");
        gen_test_all255();
        run_one_test(2);

        $display("============================================");
        $display("  Test 3: Random");
        $display("============================================");
        gen_test_random();
        run_one_test(3);

        $display("============================================");
        $display("  All 3 tests complete.");
        $display("============================================");
        $finish;
    end

    initial begin
        #5000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
