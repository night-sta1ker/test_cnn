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
    input  wire                     clk,
    input  wire                     rst_n,

    // ------------------------------------------------------------------------
    // Matrix size configuration
    //   Computes C[M x N] = A[M x K] * B[K x N].
    // ------------------------------------------------------------------------
    input  wire                     size_ld,
    input  wire [15:0]              m_size,
    input  wire [15:0]              k_size,
    input  wire [15:0]              n_size,
    input  wire [31:0]              in_act_base_addr,
    input  wire [31:0]              in_wgt_base_addr,

    // ------------------------------------------------------------------------
    // Activation stream from upstream buffer/loader
    //   Input layout is consumed by trans_act and converted to array lanes.
    // ------------------------------------------------------------------------
    input  wire [4*DATA_W-1:0]      act_stream_in,
    input  wire                     act_stream_valid,
    output reg                      act_stream_ready,
    output reg  [31:0]              act_base_addr,
    output reg                      act_base_valid,

    // ------------------------------------------------------------------------
    // Weight stream from upstream buffer/loader
    //   One N tile is requested at a time. n_base marks the tile column base.
    // ------------------------------------------------------------------------
    input  wire [4*DATA_W-1:0]      wgt_stream_in,
    input  wire                     wgt_stream_valid,
    output reg                      wgt_stream_ready,
    output reg  [31:0]              wgt_base_addr,
    output reg                      wgt_base_valid,

    // ------------------------------------------------------------------------
    // Systolic array drive signals
    // ------------------------------------------------------------------------
    output wire                     en,
    output reg                      clear,
    output reg [4*DATA_W-1:0]       act_in,
    output reg [3:0]               wgt_ld,
    output reg [4*DATA_W-1:0]      wgt_in,

    // ------------------------------------------------------------------------
    // Psum control and result path
    // ------------------------------------------------------------------------
    output reg  [15:0]              psum_zero,
    output reg  [1:0]               psum_out_sel,
    input  wire [4*ACC_W-1:0]       psum_out,
    output wire [MAX_M_SIZE*MAX_N_SIZE*ACC_W-1:0] c_out_flat
);
    // ------------------------------------------------------------------------
    // group counter, from 0 count to n/4, m/4
    // ------------------------------------------------------------------------
    reg [13:0] act_group_cnt;
    reg [13:0] wgt_group_cnt;



    wire act_stream_en = act_stream_valid && act_stream_ready;
    wire wgt_stream_en = wgt_stream_valid && wgt_stream_ready;
    wire act_out_en;
    wire wgt_out_en;
    reg  act_start_flag;
    reg  wgt_start_flag;
    reg  hold; 
    reg  hold_r;
    reg  wgt_start_first;
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
    reg [1:0] state;
    reg [15:0] m_size_r;
    reg [15:0] n_size_r;
    reg [15:0] k_size_r;
    wire act_done;
    wire wgt_done;
    reg [15:0] act_k_cnt;
    reg [15:0] wgt_k_cnt;
    reg [15:0] act_m_cnt;
    reg group_read_done;
    reg act_all_read_done;
    reg wgt_all_read_done;

    localparam S_IDLE   = 2'b00;
    localparam S_STREAM = 2'b01;
    localparam S_DONE   = 2'b10;

    // ------------------------------------------------------------------------act_in 
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
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
        else if(size_ld && (state == S_IDLE))begin
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
        else if(hold)begin
        end
        else if(!hold && (state == S_STREAM) && act_stream_en)begin
            act_lane0 <=  act_stream_in[DATA_W-1:0];
            act_lane1[0] <=  act_lane1[1];
            act_lane1[1] <=  act_stream_in[2*DATA_W-1:DATA_W];
            act_lane2[0] <=  act_lane2[1];
            act_lane2[1] <=  act_lane2[2];
            act_lane2[2] <=  act_stream_in[3*DATA_W-1:2*DATA_W];
            act_lane3[0] <=  act_lane3[1];
            act_lane3[1] <=  act_lane3[2];
            act_lane3[2] <=  act_lane3[3];
            act_lane3[3] <=  act_stream_in[4*DATA_W-1:3*DATA_W];
            act_start_flag <= 1'b1;
            act_remain_cnt <= 3'b0;
        end
        else if (!hold && (state == S_STREAM) && !act_stream_en) begin
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
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            lane_sequen <= 'b0;
            act_in <= 'b0;
        end
        else if(size_ld && (state == S_IDLE))begin
            lane_sequen <= 'b0;
            act_in <= 'b0;
        end
        else if(act_base_valid && (act_group_cnt != 14'b0))begin
            lane_sequen <= 2'b01;
            act_in <= 'b0;
        end
        else if(hold)begin
        end
        else if(!hold && (state == S_STREAM))begin
            lane_sequen <= lane_sequen + 1;
                case(lane_sequen)
                    2'b00: act_in <= {act_lane1[0],act_lane0,act_lane3[0],act_lane2[0]};
                    2'b01: act_in <= {act_lane2[0],act_lane1[0],act_lane0,act_lane3[0]};
                    2'b10: act_in <= {act_lane3[0],act_lane2[0],act_lane1[0],act_lane0};
                    2'b11: act_in <= {act_lane0,act_lane3[0],act_lane2[0],act_lane1[0]};
                endcase
        end
    end



    // ------------------------------------------------------------------------wgt_in 
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
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
        else if(size_ld && (state == S_IDLE))begin
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
        else if(hold)begin
        end
        else if(!hold && (state == S_STREAM) && wgt_stream_en)begin
            wgt_lane0 <=  wgt_stream_in[DATA_W-1:0];
            wgt_lane1[0] <=  wgt_lane1[1];
            wgt_lane1[1] <=  wgt_stream_in[2*DATA_W-1:DATA_W];
            wgt_lane2[0] <=  wgt_lane2[1];
            wgt_lane2[1] <=  wgt_lane2[2];
            wgt_lane2[2] <=  wgt_stream_in[3*DATA_W-1:2*DATA_W];
            wgt_lane3[0] <=  wgt_lane3[1];
            wgt_lane3[1] <=  wgt_lane3[2];
            wgt_lane3[2] <=  wgt_lane3[3];
            wgt_lane3[3] <=  wgt_stream_in[4*DATA_W-1:3*DATA_W];
            wgt_start_flag <= 1'b1;
            wgt_remain_cnt <= 3'b0;
        end
        else if (!hold && (state == S_STREAM) && !wgt_stream_en && (wgt_group_cnt == 14'b0)) begin
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
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            wgt_in <= 'b0;
            wgt_ld <= 4'b0001;
        end
        else if(size_ld && (state == S_IDLE))begin
            wgt_in <= 'b0;
            wgt_ld <= 4'b0001;
        end
        else if(hold)begin
        end
        else if(!hold && (state == S_STREAM))begin
                wgt_in <= {wgt_lane0,wgt_lane1[0],wgt_lane2[0],wgt_lane3[0]};
                wgt_ld <= {wgt_ld[2:0], wgt_ld[3]};
        end
    end




    // ------------------------------------------------------------------------simple control frame
    assign en = act_out_en & wgt_out_en && !hold;
    assign c_out_flat = {MAX_M_SIZE*MAX_N_SIZE*ACC_W{1'b0}};
    assign act_done = (state == S_STREAM) &&
                      act_stream_en &&
                      (act_k_cnt == (k_size_r - 1'b1)) &&
                      (act_m_cnt == (((m_size_r + 3) >> 2) - 1'b1));
    assign wgt_done = (state == S_STREAM) &&
                      wgt_stream_en &&
                      (wgt_k_cnt == (k_size_r - 1'b1));

    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            state <= S_IDLE;
            hold <= 1'b0;
            hold_r <= 1'b0;
            act_stream_ready <= 1'b0;
            wgt_stream_ready <= 1'b0;
            act_base_addr <= 32'b0;
            wgt_base_addr <= 32'b0;
            act_base_valid <= 1'b0;
            wgt_base_valid <= 1'b0;
            clear <= 1'b0;
            psum_zero <= 16'b0;
            psum_out_sel <= 2'b0;
            m_size_r <= 16'b0;
            n_size_r <= 16'b0;
            k_size_r <= 16'b0;
            act_k_cnt <= 16'b0;
            wgt_k_cnt <= 16'b0;
            act_m_cnt <= 16'b0;
            act_group_cnt <= 14'b0;
            wgt_group_cnt <= 14'b0;
            group_read_done <= 1'b0;
            act_all_read_done <= 1'b0;
            wgt_all_read_done <= 1'b0;
            wgt_start_first   <= 1'b0;
        end
        else begin
            hold_r <= hold;
            act_base_valid <= 1'b0;
            wgt_base_valid <= 1'b0;
            clear <= 1'b0;
            psum_zero <= 16'b0;
            psum_out_sel <= 2'b0;

            case(state)
                S_IDLE: begin
                    act_stream_ready <= 1'b0;
                    wgt_stream_ready <= 1'b0;

                    if(size_ld)begin
                        m_size_r <= m_size;
                        n_size_r <= n_size;
                        k_size_r <= k_size;
                        act_k_cnt <= 16'b0;
                        wgt_k_cnt <= 16'b0;
                        act_m_cnt <= 16'b0;
                        act_group_cnt <= 14'b0;
                        wgt_group_cnt <= 14'b0;
                        group_read_done <= 1'b0;
                        act_all_read_done <= 1'b0;
                        wgt_all_read_done <= 1'b0;
                        hold <= 1'b0;
                        act_base_addr <= in_act_base_addr;
                        wgt_base_addr <= in_wgt_base_addr;
                        act_base_valid <= 1'b1;
                        wgt_base_valid <= 1'b1;
                        act_stream_ready <= 1'b1;
                        wgt_stream_ready <= 1'b1;
                        state <= S_STREAM;
                        wgt_start_first   <= 1'b0;
                    end
                end

                S_STREAM: begin
                    act_stream_ready <= 1'b1;
                    wgt_stream_ready <= 1'b1;

                    if(!hold && !wgt_start_first)begin
                        wgt_start_first   <= 1'b1;
                        act_stream_ready  <= 1'b0;
                    end

                    if(hold)begin
                        act_stream_ready <= 1'b0;
                        wgt_stream_ready <= 1'b0;
                        if(act_stream_valid || wgt_stream_valid)begin
                            hold <= 1'b0;
                            if(!wgt_start_first)begin
                                wgt_start_first <= 1'b1;
                                act_stream_ready <= 1'b0;
                                wgt_stream_ready <= 1'b1;
                            end
                            else begin
                                act_stream_ready <= 1'b1;
                                wgt_stream_ready <= 1'b1;
                            end
                        end
                    end
                    else begin
                        if(group_read_done)begin
                            act_stream_ready <= 1'b0;
                            wgt_stream_ready <= 1'b0;
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
                                wgt_stream_ready <= 1'b0;
                            end
                            else begin
                                wgt_group_cnt <= wgt_group_cnt + 1'b1;
                                wgt_base_addr <= wgt_base_addr;
                                wgt_base_valid <= 1'b1;
                                hold <= 1'b1;
                                act_stream_ready <= 1'b0;
                                wgt_stream_ready <= 1'b0;
                            end
                        end
                        if(act_done)begin
                            act_k_cnt <= 16'b0;
                            wgt_k_cnt <= 16'b0;
                            act_m_cnt <= 16'b0;
                            group_read_done <= 1'b1;
                            act_stream_ready <= 1'b0;
                            wgt_stream_ready <= 1'b0;
                        end
                        else if(group_read_done && !act_out_en && !wgt_out_en)begin
                            if(act_group_cnt == (((n_size_r + 3) >> 2) - 1'b1))begin
                                act_all_read_done <= 1'b1;
                                state <= S_DONE;
                            end
                            else begin
                                act_group_cnt <= act_group_cnt + 1'b1;
                                act_base_addr <= in_act_base_addr;
                                act_base_valid <= 1'b1;
                                wgt_group_cnt <= 14'b0;
                                wgt_k_cnt <= 16'b0;
                                wgt_base_addr <= wgt_base_addr + k_size_r;
                                wgt_base_valid <= 1'b1;
                                wgt_start_first <= 1'b0;
                                group_read_done <= 1'b0;
                            end
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
                end

                S_DONE: begin
                    act_stream_ready <= 1'b0;
                    wgt_stream_ready <= 1'b0;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end


endmodule
