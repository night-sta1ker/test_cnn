`timescale 1ns / 1ps

module ProcessUnitMode0 #(
    parameter IMG_W = 28,
    parameter IMG_H = 28,
    parameter OUT_W = 26,
    parameter OUT_H = 26,
    parameter N_OC  = 16,
    parameter DATA_W = 8,
    parameter ACC_W  = 32
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start,

    input  wire [7:0]                   act,
    input  wire                         act_en,
    output wire                         act_ready,

    input  wire [7:0]                   wgt0, wgt1, wgt2, wgt3,
    input  wire [7:0]                   wgt4, wgt5, wgt6, wgt7,
    input  wire                         wgt_en,
    output wire                         wgt_ready,

    output reg  [8*ACC_W-1:0]           acc_parallel,
    output reg                          acc_en,
    output reg  [4:0]                   out_dbg_row,
    output reg  [4:0]                   out_dbg_col_base,
    output reg  [3:0]                   out_dbg_oc_group,
    output reg  [2:0]                   out_dbg_lane,
    output reg                          done
);
    localparam S_IDLE     = 3'd0;
    localparam S_LOAD     = 3'd1;
    localparam S_STEP     = 3'd2;
    localparam S_MAC      = 3'd3;
    localparam S_DONE     = 3'd6;

    localparam OC_GROUPS  = (N_OC + 7) / 8;
    localparam WGT_DEPTH  = OC_GROUPS * 9;
    localparam OC_GROUP_W = (OC_GROUPS <= 1) ? 1 :
                            (OC_GROUPS <= 2) ? 1 :
                            (OC_GROUPS <= 4) ? 2 :
                            (OC_GROUPS <= 8) ? 3 : 4;

    reg [2:0] state;

    reg [5:0] act_row;
    reg [5:0] act_col;
    reg [5:0] loaded_rows;
    reg [OC_GROUP_W-1:0] wgt_group_cnt;
    reg [3:0] wgt_k_cnt;
    reg       act_done;
    reg       wgt_loaded;

    reg [4:0] out_row;
    reg [4:0] out_col_base;
    reg [OC_GROUP_W-1:0] oc_group;
    reg [3:0] k_cnt;

    reg [7:0] line_buf [0:3][0:IMG_W-1];
    (* ram_style = "block" *) reg signed [DATA_W-1:0] wgt_bank0 [0:WGT_DEPTH-1];
    (* ram_style = "block" *) reg signed [DATA_W-1:0] wgt_bank1 [0:WGT_DEPTH-1];
    (* ram_style = "block" *) reg signed [DATA_W-1:0] wgt_bank2 [0:WGT_DEPTH-1];
    (* ram_style = "block" *) reg signed [DATA_W-1:0] wgt_bank3 [0:WGT_DEPTH-1];
    (* ram_style = "block" *) reg signed [DATA_W-1:0] wgt_bank4 [0:WGT_DEPTH-1];
    (* ram_style = "block" *) reg signed [DATA_W-1:0] wgt_bank5 [0:WGT_DEPTH-1];
    (* ram_style = "block" *) reg signed [DATA_W-1:0] wgt_bank6 [0:WGT_DEPTH-1];
    (* ram_style = "block" *) reg signed [DATA_W-1:0] wgt_bank7 [0:WGT_DEPTH-1];
    reg signed [DATA_W-1:0] wgt_q0;
    reg signed [DATA_W-1:0] wgt_q1;
    reg signed [DATA_W-1:0] wgt_q2;
    reg signed [DATA_W-1:0] wgt_q3;
    reg signed [DATA_W-1:0] wgt_q4;
    reg signed [DATA_W-1:0] wgt_q5;
    reg signed [DATA_W-1:0] wgt_q6;
    reg signed [DATA_W-1:0] wgt_q7;
    reg signed [8*ACC_W-1:0] out_buf0 [0:7];
    reg signed [8*ACC_W-1:0] out_buf1 [0:7];
    reg [7:0]  out_buf0_valid_mask;
    reg [7:0]  out_buf1_valid_mask;
    reg [4:0]  out_buf0_row;
    reg [4:0]  out_buf1_row;
    reg [4:0]  out_buf0_col_base;
    reg [4:0]  out_buf1_col_base;
    reg [3:0] out_buf0_oc_group;
    reg [3:0] out_buf1_oc_group;
    reg        out_buf0_valid;
    reg        out_buf1_valid;
    reg        cap_buf_sel;
    reg        out_buf_sel;
    reg        out_active;
    reg [2:0]  out_buf_lane;
    reg        capture_req;
    reg [4:0]  capture_row;
    reg [4:0]  capture_col_base;
    reg [3:0] capture_oc_group;
    wire       out_start_buf0;
    wire       out_start_buf1;

    wire mac_clear;
    wire mac_en;
    reg [8*DATA_W-1:0] a_vec;
    reg signed [8*DATA_W-1:0] b_vec;
    wire signed [64*ACC_W-1:0] acc_mat;

    MacArray8x8 #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) mac_array_u (
        .clk(clk),
        .rst_n(rst_n),
        .clear(mac_clear),
        .en(mac_en),
        .a_vec(a_vec),
        .b_vec(b_vec),
        .acc_mat(acc_mat)
    );

    assign out_start_buf0 = !out_active &&
                            ((!out_buf_sel && out_buf0_valid) ||
                             (out_buf_sel && !out_buf1_valid && out_buf0_valid));
    assign out_start_buf1 = !out_active && !out_start_buf0 && out_buf1_valid;

    assign act_ready = (state != S_IDLE) && (state != S_DONE) && !act_done &&
                       (act_row < ({1'b0, out_row} + 6'd4));
    assign wgt_ready = (state == S_LOAD) && !wgt_loaded;
    assign mac_clear = (state == S_STEP);
    assign mac_en    = (state == S_MAC);

    integer lane_i;
    integer col_i;
    reg [4:0] pixel_col;
    reg [4:0] oc_base;
    reg [1:0] k_row_line;
    reg [1:0] k_col_off;
    wire [9:0] wgt_wr_addr = wgt_group_cnt * 10'd9 + wgt_k_cnt;
    wire [9:0] wgt_next_addr = oc_group * 10'd9 + k_cnt + 10'd1;
    wire [4:0] wgt_oc_base = wgt_group_cnt * 5'd8;

    always @(*) begin
        a_vec = {8*DATA_W{1'b0}};
        b_vec = {8*DATA_W{1'b0}};
        oc_base = oc_group * 5'd8;

        case (k_cnt)
            4'd0, 4'd1, 4'd2: k_row_line = out_row[1:0];
            4'd3, 4'd4, 4'd5: k_row_line = out_row[1:0] + 2'd1;
            default:          k_row_line = out_row[1:0] + 2'd2;
        endcase

        case (k_cnt)
            4'd0, 4'd3, 4'd6: k_col_off = 2'd0;
            4'd1, 4'd4, 4'd7: k_col_off = 2'd1;
            default:          k_col_off = 2'd2;
        endcase

        for (lane_i = 0; lane_i < 8; lane_i = lane_i + 1) begin
            pixel_col = out_col_base + lane_i[4:0];
            if (pixel_col < OUT_W) begin
                a_vec[lane_i*DATA_W +: DATA_W] =
                    line_buf[k_row_line][pixel_col + k_col_off];
            end
        end

        b_vec[0*DATA_W +: DATA_W] = wgt_q0;
        b_vec[1*DATA_W +: DATA_W] = wgt_q1;
        b_vec[2*DATA_W +: DATA_W] = wgt_q2;
        b_vec[3*DATA_W +: DATA_W] = wgt_q3;
        b_vec[4*DATA_W +: DATA_W] = wgt_q4;
        b_vec[5*DATA_W +: DATA_W] = wgt_q5;
        b_vec[6*DATA_W +: DATA_W] = wgt_q6;
        b_vec[7*DATA_W +: DATA_W] = wgt_q7;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            act_row        <= 6'd0;
            act_col        <= 6'd0;
            loaded_rows    <= 6'd0;
            wgt_group_cnt  <= {OC_GROUP_W{1'b0}};
            wgt_k_cnt      <= 4'd0;
            act_done       <= 1'b0;
            wgt_loaded     <= 1'b0;
            out_row        <= 5'd0;
            out_col_base   <= 5'd0;
            oc_group       <= {OC_GROUP_W{1'b0}};
            k_cnt          <= 4'd0;
            wgt_q0         <= {DATA_W{1'b0}};
            wgt_q1         <= {DATA_W{1'b0}};
            wgt_q2         <= {DATA_W{1'b0}};
            wgt_q3         <= {DATA_W{1'b0}};
            wgt_q4         <= {DATA_W{1'b0}};
            wgt_q5         <= {DATA_W{1'b0}};
            wgt_q6         <= {DATA_W{1'b0}};
            wgt_q7         <= {DATA_W{1'b0}};
            capture_req    <= 1'b0;
            capture_row    <= 5'd0;
            capture_col_base <= 5'd0;
            capture_oc_group <= {OC_GROUP_W{1'b0}};
            done           <= 1'b0;
        end else begin
            capture_req <= 1'b0;

            if (act_en && act_ready) begin
                line_buf[act_row[1:0]][act_col] <= act;
                if (act_col == IMG_W-1) begin
                    act_col <= 6'd0;
                    loaded_rows <= act_row + 6'd1;
                    if (act_row == IMG_H-1)
                        act_done <= 1'b1;
                    else
                        act_row <= act_row + 6'd1;
                end else begin
                    act_col <= act_col + 6'd1;
                end
            end

            case (state)
                S_IDLE: begin
                    done          <= 1'b0;
                    act_row       <= 6'd0;
                    act_col       <= 6'd0;
                    loaded_rows   <= 6'd0;
                    wgt_group_cnt <= {OC_GROUP_W{1'b0}};
                    wgt_k_cnt     <= 4'd0;
                    act_done      <= 1'b0;
                    wgt_loaded    <= 1'b0;
                    if (start)
                        state <= S_LOAD;
                end

                S_LOAD: begin
                    if (wgt_en && !wgt_loaded) begin
                        wgt_bank0[wgt_wr_addr] <= wgt0;
                        wgt_bank1[wgt_wr_addr] <= wgt1;
                        wgt_bank2[wgt_wr_addr] <= wgt2;
                        wgt_bank3[wgt_wr_addr] <= wgt3;
                        wgt_bank4[wgt_wr_addr] <= wgt4;
                        wgt_bank5[wgt_wr_addr] <= wgt5;
                        wgt_bank6[wgt_wr_addr] <= wgt6;
                        wgt_bank7[wgt_wr_addr] <= wgt7;
                        if (wgt_k_cnt == 4'd8) begin
                            wgt_k_cnt <= 4'd0;
                            if (wgt_group_cnt == OC_GROUPS-1)
                                wgt_loaded <= 1'b1;
                            else
                                wgt_group_cnt <= wgt_group_cnt + 1'b1;
                        end else begin
                            wgt_k_cnt <= wgt_k_cnt + 4'd1;
                        end
                    end

                    if (wgt_loaded && (loaded_rows >= 6'd3)) begin
                        out_row      <= 5'd0;
                        out_col_base <= 5'd0;
                        oc_group     <= {OC_GROUP_W{1'b0}};
                        state        <= S_STEP;
                    end
                end

                S_STEP: begin
                    if (loaded_rows >= ({1'b0, out_row} + 6'd3)) begin
                        k_cnt <= 4'd0;
                        wgt_q0 <= wgt_bank0[oc_group * 10'd9];
                        wgt_q1 <= wgt_bank1[oc_group * 10'd9];
                        wgt_q2 <= wgt_bank2[oc_group * 10'd9];
                        wgt_q3 <= wgt_bank3[oc_group * 10'd9];
                        wgt_q4 <= wgt_bank4[oc_group * 10'd9];
                        wgt_q5 <= wgt_bank5[oc_group * 10'd9];
                        wgt_q6 <= wgt_bank6[oc_group * 10'd9];
                        wgt_q7 <= wgt_bank7[oc_group * 10'd9];
                        state <= S_MAC;
                    end
                end

                S_MAC: begin
                    if (k_cnt == 4'd8) begin
                        capture_req      <= 1'b1;
                        capture_row      <= out_row;
                        capture_col_base <= out_col_base;
                        capture_oc_group <= oc_group;

                        if (oc_group != OC_GROUPS-1) begin
                            oc_group <= oc_group + 1'b1;
                            state    <= S_STEP;
                        end else begin
                            oc_group <= {OC_GROUP_W{1'b0}};
                            if (out_col_base + 5'd8 < OUT_W) begin
                                out_col_base <= out_col_base + 5'd8;
                                state        <= S_STEP;
                            end else begin
                                out_col_base <= 5'd0;
                                if (out_row == OUT_H-1) begin
                                    state <= S_DONE;
                                end else begin
                                    out_row <= out_row + 5'd1;
                                    state   <= S_STEP;
                                end
                            end
                        end
                    end else begin
                        wgt_q0 <= wgt_bank0[wgt_next_addr];
                        wgt_q1 <= wgt_bank1[wgt_next_addr];
                        wgt_q2 <= wgt_bank2[wgt_next_addr];
                        wgt_q3 <= wgt_bank3[wgt_next_addr];
                        wgt_q4 <= wgt_bank4[wgt_next_addr];
                        wgt_q5 <= wgt_bank5[wgt_next_addr];
                        wgt_q6 <= wgt_bank6[wgt_next_addr];
                        wgt_q7 <= wgt_bank7[wgt_next_addr];
                        k_cnt <= k_cnt + 4'd1;
                    end
                end

                S_DONE: begin
                    if (!out_active && !out_buf0_valid && !out_buf1_valid && !capture_req) begin
                        done  <= 1'b1;
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_buf0_valid <= 1'b0;
            out_buf1_valid <= 1'b0;
            cap_buf_sel    <= 1'b0;
            out_buf_sel    <= 1'b0;
            out_active     <= 1'b0;
            out_buf_lane   <= 3'd0;
            out_dbg_row    <= 5'd0;
            out_dbg_col_base <= 5'd0;
            out_dbg_oc_group <= {OC_GROUP_W{1'b0}};
            out_dbg_lane   <= 3'd0;
            acc_parallel   <= {8*ACC_W{1'b0}};
            acc_en         <= 1'b0;
        end else begin
            acc_en <= 1'b0;

            if (out_active) begin
                if (!out_buf_sel) begin
                    acc_parallel <= out_buf0[out_buf_lane];
                    acc_en <= out_buf0_valid_mask[out_buf_lane];
                    out_dbg_row <= out_buf0_row;
                    out_dbg_col_base <= out_buf0_col_base;
                    out_dbg_oc_group <= out_buf0_oc_group;
                    out_dbg_lane <= out_buf_lane;
                    if (out_buf_lane == 3'd7) begin
                        out_active <= 1'b0;
                        out_buf0_valid <= 1'b0;
                        out_buf_sel <= 1'b1;
                        out_buf_lane <= 3'd0;
                    end else begin
                        out_buf_lane <= out_buf_lane + 3'd1;
                    end
                end else begin
                    acc_parallel <= out_buf1[out_buf_lane];
                    acc_en <= out_buf1_valid_mask[out_buf_lane];
                    out_dbg_row <= out_buf1_row;
                    out_dbg_col_base <= out_buf1_col_base;
                    out_dbg_oc_group <= out_buf1_oc_group;
                    out_dbg_lane <= out_buf_lane;
                    if (out_buf_lane == 3'd7) begin
                        out_active <= 1'b0;
                        out_buf1_valid <= 1'b0;
                        out_buf_sel <= 1'b0;
                        out_buf_lane <= 3'd0;
                    end else begin
                        out_buf_lane <= out_buf_lane + 3'd1;
                    end
                end
            end else if (out_start_buf0) begin
                out_buf_sel <= 1'b0;
                out_active <= 1'b1;
                out_buf_lane <= 3'd1;
                acc_parallel <= out_buf0[0];
                acc_en <= out_buf0_valid_mask[0];
                out_dbg_row <= out_buf0_row;
                out_dbg_col_base <= out_buf0_col_base;
                out_dbg_oc_group <= out_buf0_oc_group;
                out_dbg_lane <= 3'd0;
            end else if (out_start_buf1) begin
                out_buf_sel <= 1'b1;
                out_active <= 1'b1;
                out_buf_lane <= 3'd1;
                acc_parallel <= out_buf1[0];
                acc_en <= out_buf1_valid_mask[0];
                out_dbg_row <= out_buf1_row;
                out_dbg_col_base <= out_buf1_col_base;
                out_dbg_oc_group <= out_buf1_oc_group;
                out_dbg_lane <= 3'd0;
            end

            if (capture_req) begin
                if (!cap_buf_sel) begin
                    out_buf0[0] <= {
                        acc_mat[(0*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf0[1] <= {
                        acc_mat[(1*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf0[2] <= {
                        acc_mat[(2*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf0[3] <= {
                        acc_mat[(3*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf0[4] <= {
                        acc_mat[(4*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf0[5] <= {
                        acc_mat[(5*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf0[6] <= {
                        acc_mat[(6*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf0[7] <= {
                        acc_mat[(7*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf0_valid_mask <= {
                        (capture_col_base + 5'd7) < OUT_W,
                        (capture_col_base + 5'd6) < OUT_W,
                        (capture_col_base + 5'd5) < OUT_W,
                        (capture_col_base + 5'd4) < OUT_W,
                        (capture_col_base + 5'd3) < OUT_W,
                        (capture_col_base + 5'd2) < OUT_W,
                        (capture_col_base + 5'd1) < OUT_W,
                        (capture_col_base + 5'd0) < OUT_W
                    };
                    out_buf0_row <= capture_row;
                    out_buf0_col_base <= capture_col_base;
                    out_buf0_oc_group <= capture_oc_group;
                    out_buf0_valid <= 1'b1;
                end else begin
                    out_buf1[0] <= {
                        acc_mat[(0*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(0*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf1[1] <= {
                        acc_mat[(1*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(1*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf1[2] <= {
                        acc_mat[(2*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(2*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf1[3] <= {
                        acc_mat[(3*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(3*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf1[4] <= {
                        acc_mat[(4*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(4*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf1[5] <= {
                        acc_mat[(5*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(5*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf1[6] <= {
                        acc_mat[(6*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(6*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf1[7] <= {
                        acc_mat[(7*8 + 7)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 6)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 5)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 4)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 3)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 2)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 1)*ACC_W +: ACC_W],
                        acc_mat[(7*8 + 0)*ACC_W +: ACC_W]
                    };
                    out_buf1_valid_mask <= {
                        (capture_col_base + 5'd7) < OUT_W,
                        (capture_col_base + 5'd6) < OUT_W,
                        (capture_col_base + 5'd5) < OUT_W,
                        (capture_col_base + 5'd4) < OUT_W,
                        (capture_col_base + 5'd3) < OUT_W,
                        (capture_col_base + 5'd2) < OUT_W,
                        (capture_col_base + 5'd1) < OUT_W,
                        (capture_col_base + 5'd0) < OUT_W
                    };
                    out_buf1_row <= capture_row;
                    out_buf1_col_base <= capture_col_base;
                    out_buf1_oc_group <= capture_oc_group;
                    out_buf1_valid <= 1'b1;
                end
                cap_buf_sel <= ~cap_buf_sel;
            end
        end
    end
endmodule
