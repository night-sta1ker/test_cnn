`timescale 1ns / 1ps

module sequencer_tb;
    localparam DATA_W = 8;
    localparam ACC_W  = 32;

    reg clk;
    reg rst_n;
    integer cycle;

    reg                      size_ld;
    reg [15:0]               m_size;
    reg [15:0]               k_size;
    reg [15:0]               n_size;
    reg [31:0]               in_act_base_addr;
    reg [31:0]               in_wgt_base_addr;

    reg [4*DATA_W-1:0]       act_stream_in;
    reg                      act_stream_valid;
    wire                     act_stream_ready;
    wire [31:0]              act_base_addr;
    wire                     act_base_valid;

    reg [4*DATA_W-1:0]       wgt_stream_in;
    reg                      wgt_stream_valid;
    wire                     wgt_stream_ready;
    wire [31:0]              wgt_base_addr;
    wire                     wgt_base_valid;

    wire                     en;
    wire                     clear;
    wire [4*DATA_W-1:0]      act_in;
    wire [3:0]               wgt_ld;
    wire [4*DATA_W-1:0]      wgt_in;

    wire [15:0]              psum_zero;
    wire [1:0]               psum_out_sel;
    reg [4*ACC_W-1:0]        psum_out;
    wire [4*4*ACC_W-1:0]     c_out_flat;
    reg [4*DATA_W-1:0]       act_sram [0:255];
    reg [31:0]               act_rd_addr;
    reg [15:0]               act_words_left;
    reg                      act_loader_busy;
    reg [4*DATA_W-1:0]       wgt_sram [0:255];
    reg [31:0]               wgt_rd_addr;
    reg [15:0]               wgt_words_left;
    reg                      wgt_loader_busy;
    integer                  init_i;
    integer                  current_case;

    Sequencer #(
        .DATA_W(DATA_W),
        .ACC_W (ACC_W),
        .MAX_M_SIZE(4),
        .MAX_K_SIZE(4),
        .MAX_N_SIZE(4)
    ) dut (
        .clk              (clk),
        .rst_n            (rst_n),
        .size_ld          (size_ld),
        .m_size           (m_size),
        .k_size           (k_size),
        .n_size           (n_size),
        .in_act_base_addr (in_act_base_addr),
        .in_wgt_base_addr (in_wgt_base_addr),
        .act_stream_in    (act_stream_in),
        .act_stream_valid (act_stream_valid),
        .act_stream_ready (act_stream_ready),
        .act_base_addr    (act_base_addr),
        .act_base_valid   (act_base_valid),
        .wgt_stream_in    (wgt_stream_in),
        .wgt_stream_valid (wgt_stream_valid),
        .wgt_stream_ready (wgt_stream_ready),
        .wgt_base_addr    (wgt_base_addr),
        .wgt_base_valid   (wgt_base_valid),
        .en               (en),
        .clear            (clear),
        .act_in           (act_in),
        .wgt_ld           (wgt_ld),
        .wgt_in           (wgt_in),
        .psum_zero        (psum_zero),
        .psum_out_sel     (psum_out_sel),
        .psum_out         (psum_out),
        .c_out_flat       (c_out_flat)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle <= 0;
        else
            cycle <= cycle + 1;
    end

    task wait_done_state;
        integer timeout;
        begin
            timeout = 0;
            while (dut.state != 2'd2 && timeout < 220) begin
                timeout = timeout + 1;
                @(negedge clk);
            end

            if (dut.state != 2'd2) begin
                $display("ERROR: case=%0d timeout waiting for S_DONE at t=%0t", current_case, $time);
                $finish;
            end

            $display("DONE: case=%0d reached S_DONE at cycle=%0d", current_case, cycle);
        end
    endtask

    task clear_sram;
        begin
            for (init_i = 0; init_i < 256; init_i = init_i + 1) begin
                act_sram[init_i] = {4*DATA_W{1'b0}};
                wgt_sram[init_i] = {4*DATA_W{1'b0}};
            end
        end
    endtask

    task load_stream_case;
        input [15:0] tm;
        input [15:0] tk;
        input [15:0] tn;
        input [31:0] act_base;
        input [31:0] wgt_base;
        integer m_blk;
        integer n_blk;
        integer row;
        integer col;
        integer lane_row;
        integer lane_col;
        reg [DATA_W-1:0] v0;
        reg [DATA_W-1:0] v1;
        reg [DATA_W-1:0] v2;
        reg [DATA_W-1:0] v3;
        begin
            clear_sram();

            for (m_blk = 0; m_blk < ((tm + 3) >> 2); m_blk = m_blk + 1) begin
                for (col = 0; col < tk; col = col + 1) begin
                    lane_row = m_blk * 4;
                    v0 = (lane_row + 0 < tm) ? ((lane_row + 0) * tk + col + 1) : 0;
                    v1 = (lane_row + 1 < tm) ? ((lane_row + 1) * tk + col + 1) : 0;
                    v2 = (lane_row + 2 < tm) ? ((lane_row + 2) * tk + col + 1) : 0;
                    v3 = (lane_row + 3 < tm) ? ((lane_row + 3) * tk + col + 1) : 0;
                    act_sram[act_base + m_blk * tk + col] = {v3, v2, v1, v0};
                end
            end

            for (n_blk = 0; n_blk < ((tn + 3) >> 2); n_blk = n_blk + 1) begin
                for (row = 0; row < tk; row = row + 1) begin
                    lane_col = n_blk * 4;
                    v0 = (lane_col + 0 < tn) ? (row * tn + lane_col + 1) : 0;
                    v1 = (lane_col + 1 < tn) ? (row * tn + lane_col + 2) : 0;
                    v2 = (lane_col + 2 < tn) ? (row * tn + lane_col + 3) : 0;
                    v3 = (lane_col + 3 < tn) ? (row * tn + lane_col + 4) : 0;
                    wgt_sram[wgt_base + n_blk * tk + row] = {v3, v2, v1, v0};
                end
            end
        end
    endtask

    task print_matrices;
        input [15:0] tm;
        input [15:0] tk;
        input [15:0] tn;
        integer row;
        integer col;
        begin
            $display("A matrix:");
            for (row = 0; row < tm; row = row + 1) begin
                $write("  ");
                for (col = 0; col < tk; col = col + 1)
                    $write("%0d ", row * tk + col + 1);
                $write("\n");
            end

            $display("B matrix:");
            for (row = 0; row < tk; row = row + 1) begin
                $write("  ");
                for (col = 0; col < tn; col = col + 1)
                    $write("%0d ", row * tn + col + 1);
                $write("\n");
            end
        end
    endtask

    task run_case;
        input integer case_id;
        input [15:0] tm;
        input [15:0] tk;
        input [15:0] tn;
        input [31:0] act_base;
        input [31:0] wgt_base;
        begin
            current_case = case_id;
            load_stream_case(tm, tk, tn, act_base, wgt_base);

            rst_n = 1'b0;
            size_ld = 1'b0;
            m_size = 16'd0;
            k_size = 16'd0;
            n_size = 16'd0;
            in_act_base_addr = act_base;
            in_wgt_base_addr = wgt_base;
            repeat (4) @(negedge clk);
            rst_n = 1'b1;

            @(negedge clk);
            m_size = tm;
            k_size = tk;
            n_size = tn;
            size_ld = 1'b1;
            $display("BEGIN: case=%0d A=%0dx%0d B=%0dx%0d", case_id, tm, tk, tk, tn);
            print_matrices(tm, tk, tn);
            @(negedge clk);
            size_ld = 1'b0;

            wait_done_state();
            repeat (4) @(negedge clk);
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_stream_valid <= 1'b0;
            act_stream_in <= {4*DATA_W{1'b0}};
            act_rd_addr <= 32'b0;
            act_words_left <= 16'b0;
            act_loader_busy <= 1'b0;
        end else if (act_base_valid) begin
            act_rd_addr <= act_base_addr;
            act_words_left <= k_size * ((m_size + 16'd3) >> 2);
            act_stream_in <= act_sram[act_base_addr];
            act_stream_valid <= 1'b1;
            act_loader_busy <= 1'b1;
        end else if (act_stream_valid && act_stream_ready) begin
            if (act_words_left == 16'd1) begin
                act_stream_valid <= 1'b0;
                act_stream_in <= {4*DATA_W{1'b0}};
                act_words_left <= 16'b0;
                act_loader_busy <= 1'b0;
            end else begin
                act_rd_addr <= act_rd_addr + 1'b1;
                act_words_left <= act_words_left - 1'b1;
                act_stream_in <= act_sram[act_rd_addr + 1'b1];
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wgt_stream_valid <= 1'b0;
            wgt_stream_in <= {4*DATA_W{1'b0}};
            wgt_rd_addr <= 32'b0;
            wgt_words_left <= 16'b0;
            wgt_loader_busy <= 1'b0;
        end else if (wgt_base_valid) begin
            wgt_rd_addr <= wgt_base_addr;
            wgt_words_left <= k_size;
            wgt_stream_in <= wgt_sram[wgt_base_addr];
            wgt_stream_valid <= 1'b1;
            wgt_loader_busy <= 1'b1;
        end else if (wgt_stream_valid && wgt_stream_ready) begin
            if (wgt_words_left == 16'd1) begin
                wgt_stream_valid <= 1'b0;
                wgt_stream_in <= {4*DATA_W{1'b0}};
                wgt_words_left <= 16'b0;
                wgt_loader_busy <= 1'b0;
            end else begin
                wgt_rd_addr <= wgt_rd_addr + 1'b1;
                wgt_words_left <= wgt_words_left - 1'b1;
                wgt_stream_in <= wgt_sram[wgt_rd_addr + 1'b1];
            end
        end
    end

    initial begin
        rst_n = 1'b0;
        cycle = 0;
        size_ld = 1'b0;
        m_size = 16'd0;
        k_size = 16'd0;
        n_size = 16'd0;
        in_act_base_addr = 32'd100;
        in_wgt_base_addr = 32'd200;
        psum_out = {4*ACC_W{1'b0}};
        current_case = 0;

        run_case(1, 16'd5, 16'd6, 16'd8, 32'd100, 32'd200);
        run_case(2, 16'd2, 16'd3, 16'd2, 32'd100, 32'd200);
        run_case(3, 16'd8, 16'd9, 16'd6, 32'd100, 32'd200);

        $finish;
    end

    always @(posedge clk) begin
        if (rst_n && (act_stream_valid || wgt_stream_valid || en ||
                      act_base_valid || wgt_base_valid)) begin
            $display(
                "case=%0d cycle=%0d state=%0d hold=%b ag=%0d wg=%0d ak=%0d wk=%0d am=%0d adone=%b wdone=%b all=%b aout=%b wout=%b | act v/r=%b/%b base=%b:%0d stream=%0d,%0d,%0d,%0d act_in=%0d,%0d,%0d,%0d | wgt v/r=%b/%b base=%b:%0d stream=%0d,%0d,%0d,%0d wgt_in=%0d,%0d,%0d,%0d ld=%b | en=%b",
                current_case,
                cycle,
                dut.state,
                dut.hold,
                dut.act_group_cnt,
                dut.wgt_group_cnt,
                dut.act_k_cnt,
                dut.wgt_k_cnt,
                dut.act_m_cnt,
                dut.act_done,
                dut.wgt_done,
                dut.act_all_read_done,
                dut.act_out_en,
                dut.wgt_out_en,
                act_stream_valid,
                act_stream_ready,
                act_base_valid,
                act_base_addr,
                act_stream_in[7:0],
                act_stream_in[15:8],
                act_stream_in[23:16],
                act_stream_in[31:24],
                act_in[31:24],
                act_in[23:16],
                act_in[15:8],
                act_in[7:0],
                wgt_stream_valid,
                wgt_stream_ready,
                wgt_base_valid,
                wgt_base_addr,
                wgt_stream_in[7:0],
                wgt_stream_in[15:8],
                wgt_stream_in[23:16],
                wgt_stream_in[31:24],
                wgt_in[31:24],
                wgt_in[23:16],
                wgt_in[15:8],
                wgt_in[7:0],
                wgt_ld,
                en
            );
        end
    end
endmodule
