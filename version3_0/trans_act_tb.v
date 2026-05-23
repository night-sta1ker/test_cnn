`timescale 1ns / 1ps

module trans_act_tb;

    localparam DATA_W = 16;
    localparam M_SIZE = 8;
    localparam K_SIZE = 8;
    localparam OUT_CYCLES = 19;
    localparam MAX_CYCLES = 200;

    reg clk;
    reg rst_n;
    reg clear;

    reg [4*DATA_W-1:0] in_data;
    reg in_data_valid;
    wire in_data_ready;

    wire [4*DATA_W-1:0] out_data;
    wire out_data_valid;
    reg out_data_ready;
    wire done;

    integer feed_next;
    integer out_count;
    integer errors;
    integer lane;
    integer sim_cycle;

    reg signed [DATA_W-1:0] got;
    reg signed [DATA_W-1:0] exp;

    trans_act #(
        .DATA_W(DATA_W),
        .MAX_M_SIZE(M_SIZE),
        .MAX_K_SIZE(K_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .hold(1'b0),
        .clear(clear),
        .m_size(16'd8),
        .k_size(16'd8),
        .in_data(in_data),
        .in_data_valid(in_data_valid),
        .in_data_ready(in_data_ready),
        .out_data(out_data),
        .out_data_valid(out_data_valid),
        .out_data_ready(out_data_ready),
        .done(done)
    );

    always #5 clk = ~clk;

    function signed [DATA_W-1:0] expected_value;
        input integer out_c;
        input integer out_lane;
        integer m;
        integer k;
        integer group_id;
        integer local_row;
        integer base_cycle;
        integer base_lane;
        begin
            expected_value = 0;
            for (m = 0; m < M_SIZE; m = m + 1) begin
                group_id = m / 4;
                local_row = m - group_id*4;
                base_cycle = group_id * K_SIZE;
                base_lane = (group_id * K_SIZE) % 4;

                for (k = 0; k < K_SIZE; k = k + 1) begin
                    if (((base_cycle + local_row + k) == out_c) &&
                        (((base_lane + k) % 4) == out_lane))
                        expected_value = m*K_SIZE + k + 1;
                end
            end
        end
    endfunction

    task drive_input;
        integer l;
        integer value;
        begin
            in_data = {4*DATA_W{1'b0}};
            for (l = 0; l < 4; l = l + 1) begin
                value = feed_next + l + 1;
                if (value <= M_SIZE*K_SIZE)
                    in_data[l*DATA_W +: DATA_W] = value[DATA_W-1:0];
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        clear = 1'b0;
        in_data_valid = 1'b0;
        in_data = 0;
        out_data_ready = 1'b1;
        feed_next = 0;
        out_count = 0;
        errors = 0;
        sim_cycle = 0;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        @(posedge clk);
        clear = 1'b1;
        @(posedge clk);
        clear = 1'b0;
        in_data_valid = 1'b1;
        drive_input();

        while (!done && (sim_cycle < MAX_CYCLES)) begin
            @(posedge clk);
            #1;
            sim_cycle = sim_cycle + 1;

            if (in_data_valid && in_data_ready) begin
                feed_next = feed_next + 4;
                if (feed_next >= M_SIZE*K_SIZE) begin
                    in_data_valid = 1'b0;
                    in_data = 0;
                end else begin
                    drive_input();
                end
            end

            if (out_data_valid) begin
                if (out_count >= OUT_CYCLES) begin
                    $display("ERROR extra output cycle %0d", out_count);
                    errors = errors + 1;
                end else begin
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
                end
                out_count = out_count + 1;
            end
        end

        if (!done) begin
            $display("ERROR timeout after %0d cycles", sim_cycle);
            $display("state=%0d load_m=%0d load_k=%0d out_cycle=%0d out_valid=%0d out_count=%0d",
                     dut.state, dut.load_m, dut.load_k, dut.out_cycle,
                     out_data_valid, out_count);
            errors = errors + 1;
        end

        if (out_count != OUT_CYCLES) begin
            $display("ERROR output cycles exp=%0d got=%0d", OUT_CYCLES, out_count);
            errors = errors + 1;
        end

        if (errors == 0)
            $display("TRANS_ACT_TB PASSED");
        else
            $display("TRANS_ACT_TB FAILED errors=%0d", errors);

        #20;
        $finish;
    end

endmodule
