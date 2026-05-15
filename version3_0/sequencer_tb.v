`timescale 1ns / 1ps

module sequencer_tb;

    localparam DATA_W = 16;
    localparam ACC_W  = 32;
    localparam M_SIZE = 4;
    localparam K_SIZE = 4;
    localparam N_SIZE = 4;

    reg clk;
    reg rst_n;
    reg valid;
    wire ready;

    reg [4*DATA_W-1:0] act_stream_in;
    reg                act_stream_valid;
    wire               act_stream_ready;

    reg [4*DATA_W-1:0] wgt_stream_in;
    reg                wgt_stream_valid;
    wire               wgt_stream_ready;

    wire [3:0]              wgt_ld;
    wire [4*DATA_W-1:0]     wgt_in;
    wire                    en;
    wire                    clear;
    wire [4*DATA_W-1:0]     act_in;
    wire [15:0]             psum_zero;
    wire [1:0]              psum_out_sel;
    reg  [4*ACC_W-1:0]      psum_out;
    wire                    done;
    wire [31:0]             cycle;
    wire [M_SIZE*N_SIZE*ACC_W-1:0] c_out_flat;

    integer act_next;
    integer wgt_row;
    integer sim_cycle;
    integer lane;

    reg signed [DATA_W-1:0] lane_val;

    Sequencer #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MAX_M_SIZE(M_SIZE),
        .MAX_K_SIZE(K_SIZE),
        .MAX_N_SIZE(N_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid(valid),
        .ready(ready),
        .size_ld(1'b0),
        .m_size(16'd4),
        .k_size(16'd4),
        .n_size(16'd4),
        .act_stream_in(act_stream_in),
        .act_stream_valid(act_stream_valid),
        .act_stream_ready(act_stream_ready),
        .wgt_stream_in(wgt_stream_in),
        .wgt_stream_valid(wgt_stream_valid),
        .wgt_stream_ready(wgt_stream_ready),
        .wgt_ld(wgt_ld),
        .wgt_in(wgt_in),
        .en(en),
        .clear(clear),
        .act_in(act_in),
        .psum_zero(psum_zero),
        .psum_out_sel(psum_out_sel),
        .psum_out(psum_out),
        .done(done),
        .cycle(cycle),
        .c_out_flat(c_out_flat)
    );

    always #5 clk = ~clk;

    task drive_act;
        integer l;
        integer value;
        begin
            act_stream_in = {4*DATA_W{1'b0}};
            for (l = 0; l < 4; l = l + 1) begin
                value = act_next + l + 1;
                if (value <= M_SIZE*K_SIZE)
                    act_stream_in[l*DATA_W +: DATA_W] = value[DATA_W-1:0];
            end
        end
    endtask

    task drive_wgt;
        integer l;
        integer value;
        begin
            wgt_stream_in = {4*DATA_W{1'b0}};
            for (l = 0; l < 4; l = l + 1) begin
                value = 100 + wgt_row*N_SIZE + l + 1;
                if (wgt_row < K_SIZE)
                    wgt_stream_in[l*DATA_W +: DATA_W] = value[DATA_W-1:0];
            end
        end
    endtask

    task print_vec;
        input [4*DATA_W-1:0] vec;
        begin
            for (lane = 0; lane < 4; lane = lane + 1) begin
                lane_val = $signed(vec[lane*DATA_W +: DATA_W]);
                $write(" %0d", lane_val);
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        valid = 1'b0;
        act_stream_in = 0;
        act_stream_valid = 1'b0;
        wgt_stream_in = 0;
        wgt_stream_valid = 1'b0;
        psum_out = 0;
        act_next = 0;
        wgt_row = 0;
        sim_cycle = 0;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        @(posedge clk);
        #1;
        valid = 1'b1;
        act_stream_valid = 1'b1;
        wgt_stream_valid = 1'b1;
        drive_act();
        drive_wgt();

        @(posedge clk);
        #1;
        valid = 1'b0;

        while (!done && sim_cycle < 80) begin
            @(posedge clk);
            #1;

            if (act_stream_valid && act_stream_ready) begin
                act_next = act_next + 4;
                if (act_next >= M_SIZE*K_SIZE) begin
                    act_stream_valid = 1'b0;
                    act_stream_in = 0;
                end else begin
                    drive_act();
                end
            end

            if (wgt_stream_valid && wgt_stream_ready) begin
                wgt_row = wgt_row + 1;
                if (wgt_row >= K_SIZE) begin
                    wgt_stream_valid = 1'b0;
                    wgt_stream_in = 0;
                end else begin
                    drive_wgt();
                end
            end

            $write("CYCLE %0d state=%0d en=%0b clear=%0b wgt_ld=%b", sim_cycle, dut.state, en, clear, wgt_ld);
            $write(" | WGT");
            print_vec(wgt_in);
            $write(" | ACT");
            print_vec(act_in);
            $write(" | stream_ready act=%0b wgt=%0b", act_stream_ready, wgt_stream_ready);
            $write(" | trans_valid act=%0b wgt=%0b", dut.act_trans_out_valid, dut.wgt_trans_out_valid);
            $display("");

            sim_cycle = sim_cycle + 1;
        end

        if (done)
            $display("SEQUENCER_TB DONE at cycle %0d", sim_cycle);
        else
            $display("SEQUENCER_TB TIMEOUT");

        #20;
        $finish;
    end

endmodule
