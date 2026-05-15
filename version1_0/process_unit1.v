`timescale 1ns / 1ps

// ============================================================================
// Process_unit — 16ch, 2-pass compute, 8ch/cycle output (64-bit)
//
//   PASS0: compute ch0-7  → output ch0-7  (64-bit)
//   PASS1: compute ch8-15 → output ch8-15 (64-bit)
//   Each pixel: 2 output cycles × 64-bit
// ============================================================================
module Process_unit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,

    input  wire [7:0]  act,
    input  wire        act_en,
    output wire        act_ready,

    input  wire [7:0]  wgt_pos0, wgt_pos1, wgt_pos2,
    input  wire [7:0]  wgt_pos3, wgt_pos4, wgt_pos5,
    input  wire [7:0]  wgt_pos6, wgt_pos7, wgt_pos8,
    input  wire        wgt_en,
    output wire        wgt_ready,

    input  wire signed [15:0] mult,
    input  wire signed [7:0] shift_param,

    output reg  [63:0] out_parallel,
    output reg         out_en,
    output reg         done
);

    localparam IMG_W = 28;
    localparam N_OC  = 16;
    localparam N_PE  = 8;
    localparam OUT_W = 26;
    localparam OUT_H = 26;

    localparam S_IDLE      = 2'd0;
    localparam S_LOAD_INIT = 2'd1;
    localparam S_STREAM    = 2'd2;
    localparam S_DONE      = 2'd3;

    localparam PH_SETUP_A = 3'd0;
    localparam PH_SETUP_B = 3'd1;
    localparam PH_MAC0    = 3'd2;
    localparam PH_Q0      = 3'd3;
    localparam PH_MAC1    = 3'd4;
    localparam PH_Q1      = 3'd5;

    reg [1:0] state;
    reg [2:0] phase;

    // Weights (16ch × 9)
    reg signed [7:0]  wgt_reg  [0:N_OC-1][0:8];
    reg signed [15:0] mult_reg [0:N_OC-1];
    reg signed [7:0]  shft_reg [0:N_OC-1];

    // Line buffers
    (* ram_style = "registers" *) reg [7:0] line_buf [0:2][0:IMG_W-1];
    reg [1:0] buf_base;

    function [1:0] buf_idx;
        input [1:0] base;
        input [1:0] offset;
        reg [2:0] s;
        begin
            s = base + offset;
            buf_idx = (s >= 3) ? s - 3 : s[1:0];
        end
    endfunction

    reg [4:0] wgt_cnt;
    reg [5:0] act_init_cnt;
    reg       wgt_done;
    reg       init_ready;

    reg [4:0] out_row;
    reg [4:0] out_col;
    reg [4:0] load_col;
    reg       pass;

    // No acc_buf — ch0-7 quantized in PASS0 directly from pe_acc

    // -----------------------------------------------------------------------
    // 3x3 window
    // -----------------------------------------------------------------------
    wire [1:0] br0 = buf_idx(buf_base, 2'd0);
    wire [1:0] br1 = buf_idx(buf_base, 2'd1);
    wire [1:0] br2 = buf_idx(buf_base, 2'd2);

    wire [7:0] win0 = line_buf[br0][out_col];
    wire [7:0] win1 = line_buf[br0][out_col + 5'd1];
    wire [7:0] win2 = line_buf[br0][out_col + 5'd2];
    wire [7:0] win3 = line_buf[br1][out_col];
    wire [7:0] win4 = line_buf[br1][out_col + 5'd1];
    wire [7:0] win5 = line_buf[br1][out_col + 5'd2];
    wire [7:0] win6 = line_buf[br2][out_col];
    wire [7:0] win7 = line_buf[br2][out_col + 5'd1];
    wire [7:0] win8 = (phase == PH_MAC0) ? act : line_buf[br2][out_col + 5'd2];

    // -----------------------------------------------------------------------
    // 8 PE instances — pass-muxed weights
    // -----------------------------------------------------------------------
    wire [3:0] ch_off [0:N_PE-1];
    assign ch_off[0] = pass ? 4'd8  : 4'd0;
    assign ch_off[1] = pass ? 4'd9  : 4'd1;
    assign ch_off[2] = pass ? 4'd10 : 4'd2;
    assign ch_off[3] = pass ? 4'd11 : 4'd3;
    assign ch_off[4] = pass ? 4'd12 : 4'd4;
    assign ch_off[5] = pass ? 4'd13 : 4'd5;
    assign ch_off[6] = pass ? 4'd14 : 4'd6;
    assign ch_off[7] = pass ? 4'd15 : 4'd7;

    wire signed [31:0] pe_acc [0:N_PE-1];
    wire signed [47:0] pe_quant_prod [0:N_PE-1];
    wire               pe_mode = (phase == PH_Q0 || phase == PH_Q1);

    wire signed [15:0] q_mult  [0:7];
    wire signed [7:0]  q_shift [0:7];
    genvar qm;
    generate
        for (qm = 0; qm < 8; qm = qm + 1) begin : QMUX
            assign q_mult[qm]  = (phase == PH_Q0) ? mult_reg[qm] : mult_reg[qm+8];
            assign q_shift[qm] = (phase == PH_Q0) ? shft_reg[qm] : shft_reg[qm+8];
        end
    endgenerate

    genvar gi;
    generate
        for (gi = 0; gi < N_PE; gi = gi + 1) begin : PE_INST
            PE pe_u (
                .clk(clk),
                .rst_n(rst_n),
                .act0(win0), .act1(win1), .act2(win2),
                .act3(win3), .act4(win4), .act5(win5),
                .act6(win6), .act7(win7), .act8(win8),
                .wgt0(wgt_reg[ch_off[gi]][0]), .wgt1(wgt_reg[ch_off[gi]][1]),
                .wgt2(wgt_reg[ch_off[gi]][2]), .wgt3(wgt_reg[ch_off[gi]][3]),
                .wgt4(wgt_reg[ch_off[gi]][4]), .wgt5(wgt_reg[ch_off[gi]][5]),
                .wgt6(wgt_reg[ch_off[gi]][6]), .wgt7(wgt_reg[ch_off[gi]][7]),
                .wgt8(wgt_reg[ch_off[gi]][8]),
                .quant_mult(q_mult[gi]),
                .mode(pe_mode),
                .acc(pe_acc[gi]),
                .quant_prod(pe_quant_prod[gi])
            );
        end
    endgenerate

    // -----------------------------------------------------------------------
    // 8 time-multiplexed quantizers (shared by ch0-7 and ch8-15)
    //   PASS0: pe_acc = ch0-7, params = mult[0..7] / shift[0..7]
    //   PASS1: pe_acc = ch8-15, params = mult[8..15] / shift[8..15]
    //   No acc_buf needed — each channel quantized in the cycle it's computed
    // -----------------------------------------------------------------------
    wire [7:0] q_comb [0:7];
    genvar qi;
    generate
        for (qi = 0; qi < 8; qi = qi + 1) begin : QUANT
            wire signed [7:0]  q_sum = 8'sd15 + $signed(q_shift[qi]);
            wire [5:0]         qs    = ($signed(q_sum) > 8'sd63)  ? 6'd63
                                     : ($signed(q_sum) < 8'sd0)   ? 6'd0
                                     : q_sum[5:0];
            wire signed [47:0] qv    = $signed(pe_quant_prod[qi] >>> qs);
            assign q_comb[qi] = ($signed(qv) < $signed(32'd0))   ? 8'd0
                              : ($signed(qv) > $signed(32'd255)) ? 8'd255
                              : qv[7:0];
        end
    endgenerate

    // -----------------------------------------------------------------------
    // act_ready
    // -----------------------------------------------------------------------
    assign wgt_ready = (state == S_LOAD_INIT) && !wgt_done;

    assign act_ready = (state == S_LOAD_INIT) ? 1'b1
                     : (state == S_STREAM)
                         ? (phase == PH_SETUP_A || phase == PH_SETUP_B
                         || phase == PH_Q1)
                     : 1'b0;

    // -----------------------------------------------------------------------
    // 64-bit output — q_comb is time-muxed: PASS0=ch0-7, PASS1=ch8-15
    // -----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_parallel <= 64'd0;
        end else if (state == S_STREAM && (phase == PH_Q0 || phase == PH_Q1)) begin
            out_parallel <= {
                q_comb[7], q_comb[6], q_comb[5], q_comb[4],
                q_comb[3], q_comb[2], q_comb[1], q_comb[0]
            };
        end
    end

    // -----------------------------------------------------------------------
    // State machine
    // -----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            phase        <= PH_SETUP_A;
            out_en       <= 1'b0;
            done         <= 1'b0;
            wgt_cnt      <= 5'd0;
            act_init_cnt <= 6'd0;
            wgt_done     <= 1'b0;
            init_ready   <= 1'b0;
            out_row      <= 5'd0;
            out_col      <= 5'd0;
            load_col     <= 5'd0;
            pass         <= 1'b0;
            buf_base     <= 2'd0;
        end else begin
            out_en <= 1'b0;

            case (state)

                S_IDLE: begin
                    done        <= 1'b0;
                    wgt_cnt     <= 5'd0;
                    act_init_cnt<= 6'd0;
                    wgt_done    <= 1'b0;
                    init_ready  <= 1'b0;
                    out_row     <= 5'd0;
                    out_col     <= 5'd0;
                    load_col    <= 5'd0;
                    pass        <= 1'b0;
                    buf_base    <= 2'd0;
                    if (start) state <= S_LOAD_INIT;
                end

                S_LOAD_INIT: begin
                    if (wgt_en && !wgt_done) begin
                        wgt_reg[wgt_cnt][0] <= wgt_pos0;
                        wgt_reg[wgt_cnt][1] <= wgt_pos1;
                        wgt_reg[wgt_cnt][2] <= wgt_pos2;
                        wgt_reg[wgt_cnt][3] <= wgt_pos3;
                        wgt_reg[wgt_cnt][4] <= wgt_pos4;
                        wgt_reg[wgt_cnt][5] <= wgt_pos5;
                        wgt_reg[wgt_cnt][6] <= wgt_pos6;
                        wgt_reg[wgt_cnt][7] <= wgt_pos7;
                        wgt_reg[wgt_cnt][8] <= wgt_pos8;
                        mult_reg[wgt_cnt]   <= mult;
                        shft_reg[wgt_cnt]   <= shift_param;
                        if (wgt_cnt == 5'd15)
                            wgt_done <= 1'b1;
                        else
                            wgt_cnt <= wgt_cnt + 5'd1;
                    end

                    if (act_en) begin
                        if (act_init_cnt < 6'd28)
                            line_buf[0][act_init_cnt[4:0]] <= act;
                        else
                            line_buf[1][act_init_cnt[4:0] - 5'd28] <= act;
                        if (act_init_cnt == 6'd54)
                            init_ready <= 1'b1;
                        if (act_init_cnt < 6'd55)
                            act_init_cnt <= act_init_cnt + 6'd1;
                    end

                    if (wgt_done && init_ready) begin
                        state    <= S_STREAM;
                        phase    <= PH_SETUP_A;
                        out_row  <= 5'd0;
                        out_col  <= 5'd0;
                        load_col <= 5'd0;
                        buf_base <= 2'd0;
                    end
                end

                S_STREAM: begin
                    case (phase)

                        PH_SETUP_A: begin
                            if (act_en) begin
                                line_buf[br2][0] <= act;
                                load_col <= 5'd1;
                                phase    <= PH_SETUP_B;
                            end
                        end

                        PH_SETUP_B: begin
                            if (act_en) begin
                                line_buf[br2][1] <= act;
                                load_col <= 5'd2;
                                pass     <= 1'b0;
                                phase    <= PH_MAC0;
                            end
                        end

                        PH_MAC0: begin
                            if (act_en && load_col < IMG_W) begin
                                line_buf[br2][load_col] <= act;
                                load_col <= load_col + 5'd1;
                            end
                            phase  <= PH_Q0;
                        end

                        PH_Q0: begin
                            out_en <= 1'b1;   // low 8ch
                            pass   <= 1'b1;
                            phase  <= PH_MAC1;
                        end

                        PH_MAC1: begin
                            phase  <= PH_Q1;
                        end

                        PH_Q1: begin
                            out_en <= 1'b1;   // high 8ch

                            if (out_col == OUT_W - 1) begin
                                if (out_row == OUT_H - 1) begin
                                    state <= S_DONE;
                                end else begin
                                    out_row  <= out_row + 5'd1;
                                    buf_base <= (buf_base == 2'd2) ? 2'd0
                                                                : buf_base + 2'd1;
                                    out_col  <= 5'd0;
                                    load_col <= 5'd0;
                                    pass     <= 1'b0;
                                    phase    <= PH_SETUP_A;
                                end
                            end else begin
                                out_col <= out_col + 5'd1;
                                pass    <= 1'b0;
                                phase   <= PH_MAC0;
                            end
                        end

                        default: phase <= PH_SETUP_A;
                    endcase
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
