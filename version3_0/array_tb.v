`timescale 1ns / 1ps

module MatmulCase #(
    parameter DATA_W = 8,
    parameter ACC_W = 32,
    parameter M_SIZE = 4,
    parameter K_SIZE = 4,
    parameter N_SIZE = 4
) (
    input wire clk,
    input wire rst_n,
    output wire done,
    output wire [31:0] cycle,
    output wire [M_SIZE*N_SIZE*ACC_W-1:0] c_out_flat
);

    wire [3:0]               wgt_ld;
    wire [4*DATA_W-1:0]      wgt_in;
    wire                     en;
    wire                     clear;
    wire [4*DATA_W-1:0]      act_in;
    wire [15:0]              psum_zero;
    wire [1:0]               psum_out_sel;
    wire [4*ACC_W-1:0]       psum_out;

    Sequencer #(
        .DATA_W(DATA_W),
        .ACC_W (ACC_W),
        .MAX_M_SIZE(M_SIZE),
        .MAX_K_SIZE(K_SIZE),
        .MAX_N_SIZE(N_SIZE)
    ) seq_u (
        .clk         (clk),
        .rst_n       (rst_n),
        .m_size      (M_SIZE),
        .k_size      (K_SIZE),
        .n_size      (N_SIZE),
        .wgt_ld      (wgt_ld),
        .wgt_in      (wgt_in),
        .en          (en),
        .clear       (clear),
        .act_in      (act_in),
        .psum_zero   (psum_zero),
        .psum_out_sel(psum_out_sel),
        .psum_out    (psum_out),
        .done        (done),
        .cycle       (cycle),
        .c_out_flat  (c_out_flat)
    );

    SystolicArray4x4 #(
        .DATA_W(DATA_W),
        .ACC_W (ACC_W)
    ) array_u (
        .clk         (clk),
        .rst_n       (rst_n),
        .en          (en),
        .wgt_ld      (wgt_ld),
        .wgt_in      (wgt_in),
        .clear       (clear),
        .act_in      (act_in),
        .psum_zero   (psum_zero),
        .psum_out_sel(psum_out_sel),
        .psum_out    (psum_out)
    );

endmodule

module array_tb;

    localparam DATA_W = 8;
    localparam ACC_W  = 32;

    reg clk;
    reg rst_n;

    wire done_2x3;
    wire [31:0] cycle_2x3;
    wire [2*4*ACC_W-1:0] c_2x3;

    wire done_4x4;
    wire [31:0] cycle_4x4;
    wire [4*4*ACC_W-1:0] c_4x4;

    wire done_5x9;
    wire [31:0] cycle_5x9;
    wire [5*8*ACC_W-1:0] c_5x9;

    integer errors;
    integer i;
    integer j;
    integer k;
    integer cyc;
    integer g;
    integer col;
    integer local_m;
    integer n_block;
    integer m_group;
    integer group_base;
    integer group_start_row;
    integer group_m0;
    integer group_n0;
    integer group_rows;
    integer group_cols;
    integer k_idx;
    integer global_m;
    integer global_n;
    integer valid_loads;
    integer valid_acts;
    integer valid_zeros;
    integer valid_outputs;
    integer final_row;
    integer last_cycle_5x9;

    MatmulCase #(
        .M_SIZE(2),
        .K_SIZE(3),
        .N_SIZE(4)
    ) case_2x3_3x4 (
        .clk(clk),
        .rst_n(rst_n),
        .done(done_2x3),
        .cycle(cycle_2x3),
        .c_out_flat(c_2x3)
    );

    MatmulCase #(
        .M_SIZE(4),
        .K_SIZE(4),
        .N_SIZE(4)
    ) case_4x4_4x4 (
        .clk(clk),
        .rst_n(rst_n),
        .done(done_4x4),
        .cycle(cycle_4x4),
        .c_out_flat(c_4x4)
    );

    MatmulCase #(
        .M_SIZE(5),
        .K_SIZE(9),
        .N_SIZE(8)
    ) case_5x9_9x8 (
        .clk(clk),
        .rst_n(rst_n),
        .done(done_5x9),
        .cycle(cycle_5x9),
        .c_out_flat(c_5x9)
    );

    always #5 clk = ~clk;

    function signed [DATA_W-1:0] a_value;
        input integer row_i;
        input integer k_i;
        begin
            a_value = ((row_i*3 + k_i*5 + 1) % 9) - 4;
        end
    endfunction

    function signed [DATA_W-1:0] b_value;
        input integer k_i;
        input integer col_i;
        begin
            b_value = ((k_i*2 + col_i*3 + 2) % 7) - 3;
        end
    endfunction

    function signed [ACC_W-1:0] expected_value;
        input integer m_size;
        input integer k_size;
        input integer n_size;
        input integer row_i;
        input integer col_i;
        integer kk;
        begin
            expected_value = 0;
            for (kk = 0; kk < k_size; kk = kk + 1)
                expected_value = expected_value + a_value(row_i, kk) * b_value(kk, col_i);
        end
    endfunction

    task check_case_2x3;
        reg signed [ACC_W-1:0] got;
        reg signed [ACC_W-1:0] exp;
        begin
            for (i = 0; i < 2; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    got = $signed(c_2x3[(i*4 + j)*ACC_W +: ACC_W]);
                    exp = expected_value(2, 3, 4, i, j);
                    if (got !== exp) begin
                        $display("ERROR 2x3*3x4 C[%0d][%0d] exp=%0d got=%0d", i, j, exp, got);
                        errors = errors + 1;
                    end
                end
            end
        end
    endtask

    task check_case_4x4;
        reg signed [ACC_W-1:0] got;
        reg signed [ACC_W-1:0] exp;
        begin
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    got = $signed(c_4x4[(i*4 + j)*ACC_W +: ACC_W]);
                    exp = expected_value(4, 4, 4, i, j);
                    if (got !== exp) begin
                        $display("ERROR 4x4*4x4 C[%0d][%0d] exp=%0d got=%0d", i, j, exp, got);
                        errors = errors + 1;
                    end
                end
            end
        end
    endtask

    task check_case_5x9;
        reg signed [ACC_W-1:0] got;
        reg signed [ACC_W-1:0] exp;
        begin
            for (i = 0; i < 5; i = i + 1) begin
                for (j = 0; j < 8; j = j + 1) begin
                    got = $signed(c_5x9[(i*8 + j)*ACC_W +: ACC_W]);
                    exp = expected_value(5, 9, 8, i, j);
                    if (got !== exp) begin
                        $display("ERROR 5x9*9x8 C[%0d][%0d] exp=%0d got=%0d", i, j, exp, got);
                        errors = errors + 1;
                    end
                end
            end
        end
    endtask

    task print_5x9_cycle_report;
        input integer c;
        begin
            valid_loads = 0;
            valid_acts = 0;
            valid_zeros = 0;
            valid_outputs = 0;

            $display("CYCLE %0d", c);

            for (g = 0; g < 4; g = g + 1) begin
                n_block = g / 2;
                m_group = g - n_block*2;
                group_base = g * 9;
                group_n0 = n_block * 4;
                group_m0 = m_group * 4;
                group_start_row = group_base % 4;
                group_cols = ((8 - group_n0) >= 4) ? 4 : (8 - group_n0);
                group_rows = ((5 - group_m0) >= 4) ? 4 : (5 - group_m0);

                for (col = 0; col < group_cols; col = col + 1) begin
                    k_idx = c - group_base - col;
                    if ((k_idx >= 0) && (k_idx < 9))
                        valid_loads = valid_loads + 1;
                end
            end

            if (valid_loads != 0)
                $write("  LOAD phase=%0d", c % 4);
            else
                $write("  LOAD none");

            for (g = 0; g < 4; g = g + 1) begin
                n_block = g / 2;
                m_group = g - n_block*2;
                group_base = g * 9;
                group_n0 = n_block * 4;
                group_cols = ((8 - group_n0) >= 4) ? 4 : (8 - group_n0);
                for (col = 0; col < group_cols; col = col + 1) begin
                    k_idx = c - group_base - col;
                    if ((k_idx >= 0) && (k_idx < 9)) begin
                        global_n = group_n0 + col;
                        $write(" | c%0d<=B[%0d][%0d]=%0d(g%0d)", col, k_idx, global_n, b_value(k_idx, global_n), g);
                    end
                end
            end
            $display("");

            $write("  ACT ");
            for (g = 0; g < 4; g = g + 1) begin
                n_block = g / 2;
                m_group = g - n_block*2;
                group_base = g * 9;
                group_m0 = m_group * 4;
                group_start_row = group_base % 4;
                group_rows = ((5 - group_m0) >= 4) ? 4 : (5 - group_m0);
                for (local_m = 0; local_m < group_rows; local_m = local_m + 1) begin
                    k_idx = c - group_base - local_m - 1;
                    if ((k_idx >= 0) && (k_idx < 9)) begin
                        global_m = group_m0 + local_m;
                        $write(" | r%0d<=A[%0d][%0d]=%0d",
                               (group_start_row + k_idx) % 4, global_m, k_idx, a_value(global_m, k_idx));
                        valid_acts = valid_acts + 1;
                    end
                end
            end
            if (valid_acts == 0)
                $write("none");
            $display("");

            $write("  ZERO");
            for (g = 0; g < 4; g = g + 1) begin
                n_block = g / 2;
                m_group = g - n_block*2;
                group_base = g * 9;
                group_n0 = n_block * 4;
                group_m0 = m_group * 4;
                group_start_row = group_base % 4;
                group_cols = ((8 - group_n0) >= 4) ? 4 : (8 - group_n0);
                group_rows = ((5 - group_m0) >= 4) ? 4 : (5 - group_m0);
                for (local_m = 0; local_m < group_rows; local_m = local_m + 1) begin
                    for (col = 0; col < group_cols; col = col + 1) begin
                        if (c == (group_base + local_m + 1 + col)) begin
                            global_m = group_m0 + local_m;
                            global_n = group_n0 + col;
                            $write(" | start C[%0d][%0d] at PE[%0d][%0d]", global_m, global_n, group_start_row, col);
                            valid_zeros = valid_zeros + 1;
                        end
                    end
                end
            end
            if (valid_zeros == 0)
                $write(" none");
            $display("");

            $write("  OUT ");
            for (g = 0; g < 4; g = g + 1) begin
                n_block = g / 2;
                m_group = g - n_block*2;
                group_base = g * 9;
                group_n0 = n_block * 4;
                group_m0 = m_group * 4;
                group_start_row = group_base % 4;
                group_cols = ((8 - group_n0) >= 4) ? 4 : (8 - group_n0);
                group_rows = ((5 - group_m0) >= 4) ? 4 : (5 - group_m0);
                final_row = (group_start_row + 8) % 4;
                for (local_m = 0; local_m < group_rows; local_m = local_m + 1) begin
                    for (col = 0; col < group_cols; col = col + 1) begin
                        if (c == (group_base + local_m + 9 + col)) begin
                            global_m = group_m0 + local_m;
                            global_n = group_n0 + col;
                            $write(" | row_sel=%0d C[%0d][%0d]", final_row, global_m, global_n);
                            valid_outputs = valid_outputs + 1;
                        end
                    end
                end
            end
            if (valid_outputs == 0)
                $write("none");
            $display("");
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        errors = 0;

        repeat (3) @(negedge clk);
        #1;
        rst_n = 1'b1;

        wait (done_2x3 && done_4x4 && done_5x9);
        @(posedge clk);
        #1;

        check_case_2x3();
        check_case_4x4();
        check_case_5x9();

        last_cycle_5x9 = 39;
        $display("=== 5x9 * 9x8 schedule report ===");
        for (cyc = 0; cyc <= last_cycle_5x9; cyc = cyc + 1)
            print_5x9_cycle_report(cyc);

        $display("=== Summary ===");
        $display("2x3 * 3x4: outputs=8");
        $display("4x4 * 4x4: outputs=16");
        $display("5x9 * 9x8: outputs=40 total_cycles=%0d", last_cycle_5x9 + 1);

        if (errors == 0)
            $display("ARRAY_TB PASSED");
        else
            $display("ARRAY_TB FAILED: errors=%0d", errors);

        #20;
        $finish;
    end

endmodule
