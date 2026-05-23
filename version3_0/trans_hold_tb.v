`timescale 1ns / 1ps

module trans_hold_tb;

    localparam DATA_W = 16;

    reg clk;
    reg rst_n;

    reg act_hold;
    reg act_restart;
    reg [4*DATA_W-1:0] act_in_data;
    reg act_in_valid;
    wire act_in_ready;
    wire [4*DATA_W-1:0] act_out_data;
    wire act_out_valid;
    reg act_out_ready;
    wire act_done;

    reg wgt_hold;
    reg wgt_valid;
    reg [4*DATA_W-1:0] wgt_in_data;
    reg wgt_in_valid;
    wire wgt_in_ready;
    wire [4*DATA_W-1:0] wgt_out_data;
    wire wgt_out_valid;
    reg wgt_out_ready;
    wire wgt_block_done;
    wire wgt_done;

    reg [4*DATA_W-1:0] saved_data;
    reg saved_valid;
    reg [15:0] saved_a_load_m;
    reg [15:0] saved_a_load_k;
    reg [15:0] saved_a_out_cycle;
    reg [15:0] saved_w_load_k;
    reg [15:0] saved_w_out_idx;
    reg [1:0]  saved_w_state;

    integer i;
    integer errors;

    trans_act #(
        .DATA_W(DATA_W),
        .MAX_M_SIZE(4),
        .MAX_K_SIZE(4)
    ) act_u (
        .clk(clk),
        .rst_n(rst_n),
        .hold(act_hold),
        .clear(act_restart),
        .m_size(16'd4),
        .k_size(16'd4),
        .in_data(act_in_data),
        .in_data_valid(act_in_valid),
        .in_data_ready(act_in_ready),
        .out_data(act_out_data),
        .out_data_valid(act_out_valid),
        .out_data_ready(act_out_ready),
        .done(act_done)
    );

    trans_wgt #(
        .DATA_W(DATA_W)
    ) wgt_u (
        .clk(clk),
        .rst_n(rst_n),
        .hold(wgt_hold),
        .valid(wgt_valid),
        .k_size(16'd4),
        .n_size(16'd4),
        .in_data(wgt_in_data),
        .in_data_valid(wgt_in_valid),
        .in_data_ready(wgt_in_ready),
        .out_data(wgt_out_data),
        .out_data_valid(wgt_out_valid),
        .out_data_ready(wgt_out_ready),
        .block_done(wgt_block_done),
        .done(wgt_done)
    );

    always #5 clk = ~clk;

    task check_act_hold;
        begin
            saved_data = act_out_data;
            saved_valid = act_out_valid;
            saved_a_load_m = act_u.load_m;
            saved_a_load_k = act_u.load_k;
            saved_a_out_cycle = act_u.out_cycle;

            act_hold = 1'b1;
            act_in_valid = 1'b1;
            act_out_ready = 1'b1;

            repeat (4) begin
                @(posedge clk);
                #1;
                if (act_in_ready !== 1'b0) begin
                    $display("ACT HOLD FAIL: in_ready is not 0");
                    errors = errors + 1;
                end
                if ((act_out_data !== saved_data) ||
                    (act_out_valid !== saved_valid) ||
                    (act_u.load_m !== saved_a_load_m) ||
                    (act_u.load_k !== saved_a_load_k) ||
                    (act_u.out_cycle !== saved_a_out_cycle)) begin
                    $display("ACT HOLD FAIL: state/output changed");
                    errors = errors + 1;
                end
            end

            act_hold = 1'b0;
        end
    endtask

    task check_wgt_hold;
        begin
            saved_data = wgt_out_data;
            saved_valid = wgt_out_valid;
            saved_w_load_k = wgt_u.load_k;
            saved_w_out_idx = wgt_u.out_idx;
            saved_w_state = wgt_u.state;

            wgt_hold = 1'b1;
            wgt_in_valid = 1'b1;
            wgt_out_ready = 1'b1;

            repeat (4) begin
                @(posedge clk);
                #1;
                if (wgt_in_ready !== 1'b0) begin
                    $display("WGT HOLD FAIL: in_ready is not 0");
                    errors = errors + 1;
                end
                if ((wgt_out_data !== saved_data) ||
                    (wgt_out_valid !== saved_valid) ||
                    (wgt_u.load_k !== saved_w_load_k) ||
                    (wgt_u.out_idx !== saved_w_out_idx) ||
                    (wgt_u.state !== saved_w_state)) begin
                    $display("WGT HOLD FAIL: state/output changed");
                    errors = errors + 1;
                end
            end

            wgt_hold = 1'b0;
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        errors = 0;

        act_hold = 1'b0;
        act_restart = 1'b0;
        act_in_data = 0;
        act_in_valid = 1'b0;
        act_out_ready = 1'b1;

        wgt_hold = 1'b0;
        wgt_valid = 1'b0;
        wgt_in_data = 0;
        wgt_in_valid = 1'b0;
        wgt_out_ready = 1'b1;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        @(posedge clk);
        act_restart = 1'b1;
        wgt_valid = 1'b1;
        @(posedge clk);
        act_restart = 1'b0;
        wgt_valid = 1'b0;

        for (i = 0; i < 8; i = i + 1) begin
            @(negedge clk);
            act_in_valid = 1'b1;
            act_in_data[0*DATA_W +: DATA_W] = i*4 + 1;
            act_in_data[1*DATA_W +: DATA_W] = i*4 + 2;
            act_in_data[2*DATA_W +: DATA_W] = i*4 + 3;
            act_in_data[3*DATA_W +: DATA_W] = i*4 + 4;
            wgt_in_valid = 1'b1;
            wgt_in_data[0*DATA_W +: DATA_W] = 100 + i*4 + 1;
            wgt_in_data[1*DATA_W +: DATA_W] = 100 + i*4 + 2;
            wgt_in_data[2*DATA_W +: DATA_W] = 100 + i*4 + 3;
            wgt_in_data[3*DATA_W +: DATA_W] = 100 + i*4 + 4;
        end

        wait (act_out_valid);
        check_act_hold();

        wait (wgt_out_valid);
        check_wgt_hold();

        if (errors == 0)
            $display("TRANS_HOLD_TB PASSED");
        else
            $display("TRANS_HOLD_TB FAILED errors=%0d", errors);

        $finish;
    end

endmodule
