`timescale 1ns / 1ps

module trans_wgt_tb;

    localparam DATA_W = 16;
    localparam K_SIZE = 8;
    localparam N_SIZE = 8;
    localparam MAX_TB_CYCLES = 200;

    reg clk;
    reg rst_n;
    reg valid;

    reg [4*DATA_W-1:0] in_data;
    reg in_data_valid;
    wire in_data_ready;

    wire [4*DATA_W-1:0] out_data;
    wire out_data_valid;
    reg out_data_ready;

    wire block_done;
    wire done;

    integer feed_block;
    integer feed_row;
    integer out_count;
    integer errors;
    integer lane;
    integer tb_cycle;

    reg signed [DATA_W-1:0] got;
    reg signed [DATA_W-1:0] exp;

    trans_wgt #(
        .DATA_W(DATA_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid(valid),
        .k_size(16'd8),
        .n_size(16'd8),
        .in_data(in_data),
        .in_data_valid(in_data_valid),
        .in_data_ready(in_data_ready),
        .out_data(out_data),
        .out_data_valid(out_data_valid),
        .out_data_ready(out_data_ready),
        .block_done(block_done),
        .done(done)
    );

    always #5 clk = ~clk;

    function signed [DATA_W-1:0] matrix_value;
        input integer row_i;
        input integer col_i;
        begin
            matrix_value = row_i*N_SIZE + col_i + 1;
        end
    endfunction

    function signed [DATA_W-1:0] expected_value;
        input integer out_c;
        input integer out_lane;
        integer n_base;
        integer block_cycle;
        integer row_i;
        integer col_i;
        begin
            expected_value = 0;

            if (out_c < 11) begin
                n_base = 0;
                block_cycle = out_c;
            end else begin
                n_base = 4;
                block_cycle = out_c - 11;
            end

            col_i = n_base + out_lane;
            row_i = block_cycle - out_lane;

            if ((col_i < N_SIZE) && (row_i >= 0) && (row_i < K_SIZE))
                expected_value = matrix_value(row_i, col_i);
        end
    endfunction

    task drive_block_row;
        integer l;
        integer col_i;
        begin
            in_data = {4*DATA_W{1'b0}};
            for (l = 0; l < 4; l = l + 1) begin
                col_i = feed_block*4 + l;
                if ((feed_row < K_SIZE) && (col_i < N_SIZE))
                    in_data[l*DATA_W +: DATA_W] =
                        matrix_value(feed_row, col_i);
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        valid = 1'b0;
        in_data = 0;
        in_data_valid = 1'b0;
        out_data_ready = 1'b1;
        feed_block = 0;
        feed_row = 0;
        out_count = 0;
        errors = 0;
        tb_cycle = 0;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        @(posedge clk);
        valid = 1'b1;
        in_data_valid = 1'b1;
        drive_block_row();

        while (!done && (tb_cycle < MAX_TB_CYCLES)) begin
            @(posedge clk);
            #1;
            tb_cycle = tb_cycle + 1;

            valid = 1'b0;

            if (in_data_valid && in_data_ready) begin
                $display("IN  cycle=%0d block=%0d row=%0d data=%0d %0d %0d %0d",
                         tb_cycle, feed_block, feed_row,
                         $signed(in_data[0*DATA_W +: DATA_W]),
                         $signed(in_data[1*DATA_W +: DATA_W]),
                         $signed(in_data[2*DATA_W +: DATA_W]),
                         $signed(in_data[3*DATA_W +: DATA_W]));
                feed_row = feed_row + 1;
                if (feed_row >= K_SIZE) begin
                    in_data_valid = 1'b0;
                    in_data = 0;
                end else begin
                    drive_block_row();
                end
            end

            if (block_done) begin
                $display("BLOCK_DONE after output %0d", out_count);
                feed_block = feed_block + 1;
                feed_row = 0;
                valid = 1'b1;
                in_data_valid = 1'b1;
                drive_block_row();
            end

            if (out_data_valid) begin
                $write("OUT %0d:", out_count);
                for (lane = 0; lane < 4; lane = lane + 1) begin
                    got = $signed(out_data[lane*DATA_W +: DATA_W]);
                    exp = expected_value(out_count, lane);
                    $write(" %0d", got);
                    if (got !== exp) begin
                        $display("");
                        $display("ERROR out[%0d][%0d] exp=%0d got=%0d",
                                 out_count, lane, exp, got);
                        errors = errors + 1;
                    end
                end
                $display("");
                out_count = out_count + 1;
            end
        end

        if (!done) begin
            $display("ERROR timeout after %0d cycles", tb_cycle);
            $display("  dut.state=%0d load_k=%0d out_idx=%0d n_base=%0d block_cols=%0d",
                     dut.state, dut.load_k, dut.out_idx, dut.n_base, dut.block_cols);
            $display("  in_ready=%0b in_valid=%0b out_valid=%0b block_done=%0b done=%0b",
                     in_data_ready, in_data_valid, out_data_valid, block_done, done);
            errors = errors + 1;
        end

        if (out_count != 22) begin
            $display("ERROR output cycles exp=22 got=%0d", out_count);
            errors = errors + 1;
        end

        if (errors == 0)
            $display("TRANS_WGT_TB PASSED");
        else
            $display("TRANS_WGT_TB FAILED errors=%0d", errors);

        #20;
        $finish;
    end

endmodule
