`timescale 1ns / 1ps

// ============================================================================
// Sequencer
//
//   Top-level controller for one 4x4 systolic-array GEMM run.
//   It starts the activation/weight transform streams, feeds the array with
//   aligned 4-lane activation and weight vectors, and advances N-column tiles.
// ============================================================================

module Sequencer #(
    parameter DATA_W = 8,
    parameter ACC_W  = 32,
    parameter MAX_M_SIZE = 4,
    parameter MAX_K_SIZE = 4,
    parameter MAX_N_SIZE = 4
) (
    // ------------------------------------------------------------------------
    // Clock and reset
    // ------------------------------------------------------------------------
    input  wire                     i_clk,
    input  wire                     i_rst_n,

    // ------------------------------------------------------------------------
    // Matrix size configuration
    //   Computes C[M x N] = A[M x K] * B[K x N].
    // ------------------------------------------------------------------------
    input  wire                     i_size_ld,
    input  wire [15:0]              i_m_size,
    input  wire [15:0]              i_k_size,
    input  wire [15:0]              i_n_size,
    input  wire [31:0]              i_act_base_addr,
    input  wire [31:0]              i_wgt_base_addr,

    // ------------------------------------------------------------------------
    // Activation stream from upstream buffer/loader
    //   Input layout is consumed by trans_act and converted to array lanes.
    // ------------------------------------------------------------------------
    input  wire [4*DATA_W-1:0]      i_act_stream_data,
    input  wire                     i_act_stream_valid,
    output reg                      o_act_stream_ready,
    output reg  [31:0]              o_act_base_addr,
    output reg                      o_act_base_valid,

    // ------------------------------------------------------------------------
    // Weight stream from upstream buffer/loader
    //   One N tile is requested at a time. n_base marks the tile column base.
    // ------------------------------------------------------------------------
    input  wire [4*DATA_W-1:0]      i_wgt_stream_data,
    input  wire                     i_wgt_stream_valid,
    output reg                      o_wgt_stream_ready,
    output reg  [31:0]              o_wgt_base_addr,
    output reg                      o_wgt_base_valid,

    // ------------------------------------------------------------------------
    // Systolic array drive signals
    // ------------------------------------------------------------------------
    output wire                     o_array_en,
    output reg                      o_array_clear,
    output reg [4*DATA_W-1:0]       o_act_array_data,
    output reg [3:0]               o_wgt_array_ld,
    output reg [4*DATA_W-1:0]      o_wgt_array_data,

    // ------------------------------------------------------------------------
    // Psum control and result path
    // ------------------------------------------------------------------------
    output wire  [15:0]             o_psum_zero,
    output reg  [1:0]               o_psum_sel,
    input  wire [4*ACC_W-1:0]       i_psum_data,
    output wire [MAX_M_SIZE*MAX_N_SIZE*ACC_W-1:0] o_c_data_flat
);
    // ------------------------------------------------------------------------
    // group counter, from 0 count to n/4, m/4
    // ------------------------------------------------------------------------
    reg [13:0] act_group_cnt;
    reg [13:0] wgt_group_cnt;



    wire act_stream_en = i_act_stream_valid && o_act_stream_ready;
    wire wgt_stream_en = i_wgt_stream_valid && o_wgt_stream_ready;
    wire act_out_en;
    wire wgt_out_en;
    reg  act_start_flag;
    reg  wgt_start_flag;
    reg  hold; 
    reg [2:0] act_remain_cnt;
    reg [2:0] wgt_remain_cnt;
    reg[DATA_W-1:0] act_lane0;
    reg[DATA_W-1:0] act_lane1[1:0];
    reg[DATA_W-1:0] act_lane2[2:0];
    reg[DATA_W-1:0] act_lane3[3:0];

    reg[DATA_W-1:0] wgt_lane0;
    reg[DATA_W-1:0] wgt_lane1[1:0];
    reg[DATA_W-1:0] wgt_lane2[2:0];
    reg[DATA_W-1:0] wgt_lane3[3:0];

    reg [1:0] lane_sequen;
    reg [2:0] state;
    reg [15:0] m_size_r;
    reg [15:0] n_size_r;
    reg [15:0] k_size_r;
    wire act_done;
    wire wgt_done;
    reg [15:0] act_k_cnt;
    reg [15:0] wgt_k_cnt;
    reg [15:0] act_m_cnt;
    reg act_all_read_done;
    reg wgt_all_read_done;
    localparam S_IDLE        = 3'd0;
    localparam S_WGT_LEAD    = 3'd1;
    localparam S_STREAM      = 3'd2;
    localparam S_REPLAY_WAIT = 3'd3;
    localparam S_DRAIN       = 3'd4;
    localparam S_DONE        = 3'd6;

    wire pipe_run = ((state == S_WGT_LEAD) && wgt_stream_en) ||
                    ((state == S_STREAM) &&
                     act_stream_en &&
                     (wgt_stream_en || wgt_all_read_done)) ||
                    (state == S_DRAIN);

    // ------------------------------------------------------------------------array activation data
    always@(posedge i_clk or negedge i_rst_n)begin
        if(!i_rst_n)begin
            act_lane0 <= 'b0;
            act_lane1[0] <= 'b0;
            act_lane1[1] <= 'b0;
            act_lane2[0] <= 'b0;
            act_lane2[1] <= 'b0;
            act_lane2[2] <= 'b0;
            act_lane3[0] <= 'b0;
            act_lane3[1] <= 'b0;
            act_lane3[2] <= 'b0;
            act_lane3[3] <= 'b0;
            act_remain_cnt <= 'b0;
            act_start_flag <= 'b0;
        end
        else if(i_size_ld && (state == S_IDLE))begin
            act_lane0 <= 'b0;
            act_lane1[0] <= 'b0;
            act_lane1[1] <= 'b0;
            act_lane2[0] <= 'b0;
            act_lane2[1] <= 'b0;
            act_lane2[2] <= 'b0;
            act_lane3[0] <= 'b0;
            act_lane3[1] <= 'b0;
            act_lane3[2] <= 'b0;
            act_lane3[3] <= 'b0;
            act_remain_cnt <= 'b0;
            act_start_flag <= 'b0;
        end
        else if(!pipe_run)begin
        end
        else if(act_stream_en)begin
            act_lane0 <=  i_act_stream_data[DATA_W-1:0];
            act_lane1[0] <=  act_lane1[1];
            act_lane1[1] <=  i_act_stream_data[2*DATA_W-1:DATA_W];
            act_lane2[0] <=  act_lane2[1];
            act_lane2[1] <=  act_lane2[2];
            act_lane2[2] <=  i_act_stream_data[3*DATA_W-1:2*DATA_W];
            act_lane3[0] <=  act_lane3[1];
            act_lane3[1] <=  act_lane3[2];
            act_lane3[2] <=  act_lane3[3];
            act_lane3[3] <=  i_act_stream_data[4*DATA_W-1:3*DATA_W];
            act_start_flag <= 1'b1;
            act_remain_cnt <= 3'b0;
        end
        else begin
            act_lane0 <=  {DATA_W{1'b0}};
            act_lane1[0] <=  act_lane1[1];
            act_lane1[1] <=  {DATA_W{1'b0}};
            act_lane2[0] <=  act_lane2[1];
            act_lane2[1] <=  act_lane2[2];
            act_lane2[2] <=  {DATA_W{1'b0}};
            act_lane3[0] <=  act_lane3[1];
            act_lane3[1] <=  act_lane3[2];
            act_lane3[2] <=  act_lane3[3];
            act_lane3[3] <=  {DATA_W{1'b0}};
            if(act_remain_cnt < 3'd5)
                act_remain_cnt <= act_remain_cnt + 1;
        end
    end

    assign act_out_en = ~(act_remain_cnt == 3'd5) & act_start_flag ;



    // ------------------------------------------------------------------------act_out
    always@(posedge i_clk or negedge i_rst_n)begin
        if(!i_rst_n)begin
            lane_sequen <= 'b0;
            o_act_array_data <= 'b0;
        end
        else if(i_size_ld && (state == S_IDLE))begin
            lane_sequen <= 'b0;
            o_act_array_data <= 'b0;
        end
        else if(o_act_base_valid && (act_group_cnt != 14'b0))begin
            lane_sequen <= 2'b00;
            o_act_array_data <= 'b0;
        end
        else if(!pipe_run)begin
        end
        else begin
            lane_sequen <= lane_sequen + 1;
                case(lane_sequen)
                    2'b00: o_act_array_data <= {act_lane3[0],act_lane0,act_lane1[0],act_lane2[0]};
                    2'b01: o_act_array_data <= {act_lane0,act_lane1[0],act_lane2[0],act_lane3[0]};
                    2'b10: o_act_array_data <= {act_lane1[0],act_lane2[0],act_lane3[0],act_lane0};
                    2'b11: o_act_array_data <= {act_lane2[0],act_lane3[0],act_lane0,act_lane1[0]};
                endcase
        end
    end



    // ------------------------------------------------------------------------array weight data
    always@(posedge i_clk or negedge i_rst_n)begin
        if(!i_rst_n)begin
            wgt_lane0 <= 'b0;
            wgt_lane1[0] <= 'b0;
            wgt_lane1[1] <= 'b0;
            wgt_lane2[0] <= 'b0;
            wgt_lane2[1] <= 'b0;
            wgt_lane2[2] <= 'b0;
            wgt_lane3[0] <= 'b0;
            wgt_lane3[1] <= 'b0;
            wgt_lane3[2] <= 'b0;
            wgt_lane3[3] <= 'b0;
            wgt_remain_cnt <= 'b0;
            wgt_start_flag <= 'b0;
        end
        else if(i_size_ld && (state == S_IDLE))begin
            wgt_lane0 <= 'b0;
            wgt_lane1[0] <= 'b0;
            wgt_lane1[1] <= 'b0;
            wgt_lane2[0] <= 'b0;
            wgt_lane2[1] <= 'b0;
            wgt_lane2[2] <= 'b0;
            wgt_lane3[0] <= 'b0;
            wgt_lane3[1] <= 'b0;
            wgt_lane3[2] <= 'b0;
            wgt_lane3[3] <= 'b0;
            wgt_remain_cnt <= 'b0;
            wgt_start_flag <= 'b0;
        end
        else if(!pipe_run)begin
        end
        else if(wgt_stream_en)begin
            wgt_lane0 <=  i_wgt_stream_data[DATA_W-1:0];
            wgt_lane1[0] <=  wgt_lane1[1];
            wgt_lane1[1] <=  i_wgt_stream_data[2*DATA_W-1:DATA_W];
            wgt_lane2[0] <=  wgt_lane2[1];
            wgt_lane2[1] <=  wgt_lane2[2];
            wgt_lane2[2] <=  i_wgt_stream_data[3*DATA_W-1:2*DATA_W];
            wgt_lane3[0] <=  wgt_lane3[1];
            wgt_lane3[1] <=  wgt_lane3[2];
            wgt_lane3[2] <=  wgt_lane3[3];
            wgt_lane3[3] <=  i_wgt_stream_data[4*DATA_W-1:3*DATA_W];
            wgt_start_flag <= 1'b1;
            wgt_remain_cnt <= 3'b0;
        end
        else if (wgt_group_cnt == 14'b0) begin
            wgt_lane0 <=  {DATA_W{1'b0}};
            wgt_lane1[0] <=  wgt_lane1[1];
            wgt_lane1[1] <=  {DATA_W{1'b0}};
            wgt_lane2[0] <=  wgt_lane2[1];
            wgt_lane2[1] <=  wgt_lane2[2];
            wgt_lane2[2] <=  {DATA_W{1'b0}};
            wgt_lane3[0] <=  wgt_lane3[1];
            wgt_lane3[1] <=  wgt_lane3[2];
            wgt_lane3[2] <=  wgt_lane3[3];
            wgt_lane3[3] <=  {DATA_W{1'b0}};
            if(wgt_remain_cnt < 3'd6)
                wgt_remain_cnt <= wgt_remain_cnt + 1;
        end
    end

    assign wgt_out_en = ~(wgt_remain_cnt == 3'd6) & wgt_start_flag;



    // ------------------------------------------------------------------------wgt_out
    always@(posedge i_clk or negedge i_rst_n)begin
        if(!i_rst_n)begin
            o_wgt_array_data <= 'b0;
            o_wgt_array_ld <= 4'b0001;
        end
        else if(i_size_ld && (state == S_IDLE))begin
            o_wgt_array_data <= 'b0;
            o_wgt_array_ld <= 4'b0001;
        end
        else if(o_act_base_valid)begin
            o_wgt_array_ld <= 4'b0001;
        end
        else if(!pipe_run)begin
        end
        else begin
                o_wgt_array_data <= {wgt_lane3[0],wgt_lane2[0],wgt_lane1[0],wgt_lane0};
                if(o_array_en)
                    o_wgt_array_ld <= {o_wgt_array_ld[2:0], o_wgt_array_ld[3]};
        end
    end

    // ------------------------------------------------------------------------simple control frame
    wire array_feed_en = act_out_en & wgt_out_en && pipe_run;
    assign o_array_en = array_feed_en;
    assign o_c_data_flat = {MAX_M_SIZE*MAX_N_SIZE*ACC_W{1'b0}};
    assign act_done = (state == S_STREAM) &&
                      act_stream_en &&
                      (act_k_cnt == (k_size_r - 1'b1)) &&
                      (act_m_cnt == (((m_size_r + 3) >> 2) - 1'b1));
    assign wgt_done = (state == S_STREAM) &&
                      wgt_stream_en &&
                      (wgt_k_cnt == (k_size_r - 1'b1));

    always@(posedge i_clk or negedge i_rst_n)begin
        if(!i_rst_n)begin
            state <= S_IDLE;
            hold <= 1'b0;
            o_act_stream_ready <= 1'b0;
            o_wgt_stream_ready <= 1'b0;
            o_act_base_addr <= 32'b0;
            o_wgt_base_addr <= 32'b0;
            o_act_base_valid <= 1'b0;
            o_wgt_base_valid <= 1'b0;
            o_array_clear <= 1'b0;
            o_psum_sel <= 2'b0;
            m_size_r <= 16'b0;
            n_size_r <= 16'b0;
            k_size_r <= 16'b0;
            act_k_cnt <= 16'b0;
            wgt_k_cnt <= 16'b0;
            act_m_cnt <= 16'b0;
            act_group_cnt <= 14'b0;
            wgt_group_cnt <= 14'b0;
            act_all_read_done <= 1'b0;
            wgt_all_read_done <= 1'b0;
        end
        else begin
            o_act_base_valid <= 1'b0;
            o_wgt_base_valid <= 1'b0;
            o_array_clear <= 1'b0;

            case(state)
                S_IDLE: begin
                    o_act_stream_ready <= 1'b0;
                    o_wgt_stream_ready <= 1'b0;

                    if(i_size_ld)begin
                        m_size_r <= i_m_size;
                        n_size_r <= i_n_size;
                        k_size_r <= i_k_size;
                        act_k_cnt <= 16'b0;
                        wgt_k_cnt <= 16'b0;
                        act_m_cnt <= 16'b0;
                        act_group_cnt <= 14'b0;
                        wgt_group_cnt <= 14'b0;
                        act_all_read_done <= 1'b0;
                        wgt_all_read_done <= 1'b0;
                        hold <= 1'b0;
                        o_act_base_addr <= i_act_base_addr;
                        o_wgt_base_addr <= i_wgt_base_addr;
                        o_act_base_valid <= 1'b1;
                        o_wgt_base_valid <= 1'b1;
                        o_act_stream_ready <= 1'b0;
                        o_wgt_stream_ready <= 1'b1;
                        state <= S_WGT_LEAD;
                    end
                end

                S_WGT_LEAD: begin
                    hold <= 1'b0;
                    o_act_stream_ready <= 1'b0;
                    o_wgt_stream_ready <= 1'b1;

                    if(wgt_stream_en)begin
                        if(wgt_k_cnt == (k_size_r - 1'b1))begin
                            wgt_k_cnt <= 16'b0;
                        end
                        else begin
                            wgt_k_cnt <= wgt_k_cnt + 1'b1;
                        end
                        if(i_act_stream_valid)begin
                            o_act_stream_ready <= 1'b1;
                            o_wgt_stream_ready <= 1'b1;
                            state <= S_STREAM;
                        end
                        else begin
                            o_act_stream_ready <= 1'b0;
                            o_wgt_stream_ready <= 1'b0;
                            state <= S_REPLAY_WAIT;
                        end
                    end
                end

                S_STREAM: begin
                    hold <= 1'b0;
                    if(wgt_all_read_done)begin
                        o_act_stream_ready <= 1'b1;
                        o_wgt_stream_ready <= 1'b0;
                    end
                    else begin
                        o_act_stream_ready <= i_wgt_stream_valid;
                        o_wgt_stream_ready <= i_act_stream_valid;
                    end

                    if(wgt_stream_en)begin
                        if(wgt_k_cnt == (k_size_r - 1'b1))begin
                            wgt_k_cnt <= 16'b0;
                        end
                        else begin
                            wgt_k_cnt <= wgt_k_cnt + 1'b1;
                        end
                    end

                    if(wgt_done)begin
                        if(wgt_group_cnt == (((m_size_r + 3) >> 2) - 1'b1))begin
                            wgt_group_cnt <= 14'b0;
                            wgt_all_read_done <= 1'b1;
                            o_wgt_stream_ready <= 1'b0;
                        end
                        else begin
                            wgt_group_cnt <= wgt_group_cnt + 1'b1;
                            o_wgt_base_addr <= o_wgt_base_addr;
                            o_wgt_base_valid <= 1'b1;
                            hold <= 1'b1;
                            state <= S_REPLAY_WAIT;
                            o_act_stream_ready <= 1'b0;
                            o_wgt_stream_ready <= 1'b0;
                        end
                    end

                    if(act_done)begin
                        act_k_cnt <= 16'b0;
                        wgt_k_cnt <= 16'b0;
                        act_m_cnt <= 16'b0;
                        state <= S_DRAIN;
                        o_act_stream_ready <= 1'b0;
                        o_wgt_stream_ready <= 1'b0;
                    end
                    else if(act_stream_en)begin
                        if(act_k_cnt == (k_size_r - 1'b1))begin
                            act_k_cnt <= 16'b0;
                            act_m_cnt <= act_m_cnt + 1'b1;
                        end
                        else begin
                            act_k_cnt <= act_k_cnt + 1'b1;
                        end
                    end
                end

                S_REPLAY_WAIT: begin
                    hold <= 1'b1;
                    o_act_stream_ready <= 1'b0;
                    o_wgt_stream_ready <= 1'b0;

                    if(i_act_stream_valid && i_wgt_stream_valid)begin
                        hold <= 1'b0;
                        o_act_stream_ready <= 1'b1;
                        o_wgt_stream_ready <= 1'b1;
                        state <= S_STREAM;
                    end
                end

                S_DRAIN: begin
                    hold <= 1'b0;
                    o_act_stream_ready <= 1'b0;
                    o_wgt_stream_ready <= 1'b0;

                    if(!act_out_en && !wgt_out_en)begin
                        if(act_group_cnt == (((n_size_r + 3) >> 2) - 1'b1))begin
                            act_all_read_done <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            act_group_cnt <= act_group_cnt + 1'b1;
                            wgt_group_cnt <= 14'b0;
                            wgt_k_cnt <= 16'b0;
                            wgt_all_read_done <= 1'b0;
                            o_act_base_addr <= i_act_base_addr;
                            o_act_base_valid <= 1'b1;
                            o_wgt_base_addr <= o_wgt_base_addr + k_size_r;
                            o_wgt_base_valid <= 1'b1;
                            state <= S_WGT_LEAD;
                        end
                    end
                end

                S_DONE: begin
                    hold <= 1'b0;
                    o_act_stream_ready <= 1'b0;
                    o_wgt_stream_ready <= 1'b0;
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase

        end
    end

// ------------------------------------------------------------------------psum control 
    reg[1:0] psum_zero_sel;
    reg[3:0] psum_zero_group[3:0];
    wire psum_zero_start = (state == S_STREAM) && o_array_en &&
                           (((k_size_r == 16'd1) && act_start_flag && (act_k_cnt == 16'd0)) ||
                            ((k_size_r != 16'd1) && (act_k_cnt == 16'd1)));
    assign o_psum_zero = {psum_zero_group[3],psum_zero_group[2],psum_zero_group[1],psum_zero_group[0]};

    function [3:0] psum_zero_next;
        input [3:0] cur;
        begin
            case(cur)
                4'b0000: psum_zero_next = 4'b0000;
                4'b0001: psum_zero_next = 4'b0011;
                4'b0011: psum_zero_next = 4'b0111;
                4'b0111: psum_zero_next = 4'b1111;
                4'b1111: psum_zero_next = 4'b1110;
                4'b1110: psum_zero_next = 4'b1100;
                4'b1100: psum_zero_next = 4'b1000;
                4'b1000: psum_zero_next = 4'b0000;
                default: psum_zero_next = 4'b0000;
            endcase
        end
    endfunction

    always@(posedge i_clk or negedge i_rst_n)begin
        if(!i_rst_n)begin
            psum_zero_sel <= 'b0;
        end
        else if(i_size_ld && (state == S_IDLE))begin
            psum_zero_sel <= 2'b00;
        end
        else if(o_act_base_valid)begin
            psum_zero_sel <= 2'b00;
        end
        else if(state == S_WGT_LEAD)begin
            psum_zero_sel <= 2'b00;
        end
        else if(state == S_STREAM && o_array_en && act_k_cnt == (k_size_r - 1'b1))begin
            psum_zero_sel <= (psum_zero_sel + k_size_r)%4;
        end
        else begin
        end
    end

    always@(posedge i_clk or negedge i_rst_n)begin
        if(!i_rst_n)begin
            psum_zero_group[0] <= 4'b0000;
            psum_zero_group[1] <= 4'b0000;
            psum_zero_group[2] <= 4'b0000;
            psum_zero_group[3] <= 4'b0000;
        end
        else if((i_size_ld && (state == S_IDLE)) || o_act_base_valid)begin
            psum_zero_group[0] <= 4'b0000;
            psum_zero_group[1] <= 4'b0000;
            psum_zero_group[2] <= 4'b0000;
            psum_zero_group[3] <= 4'b0000;
        end
        else if(o_array_en)begin
            if(psum_zero_start && (psum_zero_sel == 2'd0))
                psum_zero_group[0] <= 4'b0001;
            else
                psum_zero_group[0] <= psum_zero_next(psum_zero_group[0]);

            if(psum_zero_start && (psum_zero_sel == 2'd1))
                psum_zero_group[1] <= 4'b0001;
            else
                psum_zero_group[1] <= psum_zero_next(psum_zero_group[1]);

            if(psum_zero_start && (psum_zero_sel == 2'd2))
                psum_zero_group[2] <= 4'b0001;
            else
                psum_zero_group[2] <= psum_zero_next(psum_zero_group[2]);

            if(psum_zero_start && (psum_zero_sel == 2'd3))
                psum_zero_group[3] <= 4'b0001;
            else
                psum_zero_group[3] <= psum_zero_next(psum_zero_group[3]);
        end
        else begin
            psum_zero_group[0] <= psum_zero_group[0];
            psum_zero_group[1] <= psum_zero_group[1];
            psum_zero_group[2] <= psum_zero_group[2];
            psum_zero_group[3] <= psum_zero_group[3];
        end
    end

endmodule
