`timescale 1ns / 1ps

module systolic_pe_tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg wgt_ld = 1'b0;
    reg en = 1'b0;
    reg clear = 1'b0;
    reg signed [7:0] wgt_in = 0;
    reg signed [7:0] act_in = 0;
    reg signed [31:0] psum_in = 0;
    wire signed [7:0] act_out;
    wire signed [31:0] psum_out;

    always #5 clk = ~clk;

    SystolicPE dut (
        .clk(clk), .rst_n(rst_n), .wgt_ld(wgt_ld), .wgt_in(wgt_in),
        .en(en), .clear(clear), .act_in(act_in), .psum_in(psum_in),
        .act_out(act_out), .psum_out(psum_out)
    );

    initial begin
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        en = 1'b1;
        wgt_ld = 1'b1;
        wgt_in = 8'sd2;
        act_in = 8'sd2;
        psum_in = 32'sd3;
        @(posedge clk);
        #1;
        if (psum_out !== 32'sd3)
            $fatal(1, "PE load cycle used the old weight incorrectly");

        @(negedge clk);
        wgt_ld = 1'b0;
        @(posedge clk);
        #1;
        if (psum_out !== 32'sd7)
            $fatal(1, "PE MAC mismatch data=%0d", psum_out);

        @(negedge clk);
        en = 1'b0;
        clear = 1'b1;
        @(posedge clk);
        #1;
        if (act_out !== 0 || psum_out !== 0)
            $fatal(1, "PE clear failed while stalled");

        @(negedge clk);
        clear = 1'b0;
        en = 1'b1;
        act_in = 8'sd3;
        psum_in = 32'sd1;
        @(posedge clk);
        #1;
        if (psum_out !== 32'sd7)
            $fatal(1, "PE weight changed across clear");

        $display("SYSTOLIC PE TEST PASS");
        $finish;
    end
endmodule
