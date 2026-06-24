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
    input  wire [15:0]              i_k_size,                //k should be > 4
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
    output reg  [15:0]             o_psum_zero,
    output reg  [7:0]               o_psum_sel,
    input  wire [4*ACC_W-1:0]       i_psum_data,
    output wire [3:0]               o_psum_valid
    // output wire [MAX_M_SIZE*MAX_N_SIZE*ACC_W-1:0] o_c_data_flat
);
    // ------------------------------------------------------------------------
    // Tile and stream bookkeeping
    // ------------------------------------------------------------------------
    // act_group_cnt selects the current N-column tile of C/B. It drives the
    // outer loop that advances o_wgt_base_addr after one C[M x 4] tile drains.
    reg [13:0] act_group_cnt;
    // wgt_group_cnt replays the same B tile for each M-row block of A.
    reg [13:0] wgt_group_cnt;

    // One-cycle handshake pulses for the upstream activation/weight streams.
    wire act_stream_en = i_act_stream_valid && o_act_stream_ready;
    wire wgt_stream_en = i_wgt_stream_valid && o_wgt_stream_ready;

    // Valid data is still being shifted from the local diagonal aligners into
    // o_act_array_data/o_wgt_array_data.
    wire act_out_en;
    wire wgt_out_en;

    // Set after the first accepted stream word. Used with *_remain_cnt to keep
    // shifting tail zeros until the 4x4 diagonal input window is fully drained.
    reg  act_start_flag;
    reg  wgt_start_flag;

    // Tail-drain counters for the activation/weight diagonal aligners. They
    // decide when act_out_en/wgt_out_en can deassert after stream input stops.
    reg [2:0] act_remain_cnt;
    reg [2:0] wgt_remain_cnt;

    // Activation diagonal alignment FIFOs. They transform the packed upstream
    // stream into the skewed row inputs required by o_act_array_data.
    reg[DATA_W-1:0] act_lane0;
    reg[DATA_W-1:0] act_lane1[1:0];
    reg[DATA_W-1:0] act_lane2[2:0];
    reg[DATA_W-1:0] act_lane3[3:0];

    // Weight diagonal alignment FIFOs. They feed o_wgt_array_data and rotate
    // together with o_wgt_array_ld.
    reg[DATA_W-1:0] wgt_lane0;
    reg[DATA_W-1:0] wgt_lane1[1:0];
    reg[DATA_W-1:0] wgt_lane2[2:0];
    reg[DATA_W-1:0] wgt_lane3[3:0];

    // Rotation phase used only by o_act_array_data packing.
    reg [1:0] lane_sequen;

    // Main control FSM.
    reg [2:0] state;
    reg [2:0] state_d;

    // Registered one-cycle address requests. Keeping these as pulses lets the
    // stream loaders observe a stable base address without scattering valid
    // assignments through the state/counter update block.
    reg act_base_req_q;
    reg wgt_base_req_q;

    // Latched matrix sizes from i_*_size. The sequencer uses these until DONE.
    reg [15:0] m_size_r;
    reg [15:0] n_size_r;
    reg [15:0] k_size_r;

    // End-of-block strobes for the current activation/weight stream pass.
    wire act_done;
    wire wgt_done;

    // K counters for accepted activation/weight stream words.
    reg [15:0] act_k_cnt;
    reg [15:0] wgt_k_cnt;

    // Current M-row block index for A/C. One block contains up to 4 rows.
    reg [15:0] act_m_cnt;

    // Latches that indicate the current N tile has consumed all A/B stream data.
    reg act_all_read_done;
    reg wgt_all_read_done;

    reg [1:0] drain_cnt;
    reg act_out_en_r;
    reg wgt_done_r;

    reg sel_trigger;
    reg [2:0] wgt_done_cnt;
    reg [3:0] psum_valid [1:0];
    reg [1:0] psum_valid_cnt;
    reg [4:0] psum_valid_mode [1:0];
    reg       mode_switch;
    assign o_psum_valid = (psum_valid [0] | psum_valid [1]) & {4{o_array_en}} ;

    wire wgt_group_cnt_end = (wgt_group_cnt == ((m_size_r + 3) >> 2));

    // FSM states:
    // S_IDLE        waits for i_size_ld and issues first A/B base addresses.
    // S_WGT_LEAD    accepts the first B word before starting A/B together.
    // S_STREAM      streams A/B into the array for the active tile.
    // S_REPLAY_WAIT waits until both streams are valid for the next replay.
    // S_DRAIN       keeps o_array_en running until all scheduled psums emerge.
    // S_DONE        one-cycle completion state, then returns to S_IDLE.
    localparam S_IDLE        = 3'd0;
    localparam S_WGT_LEAD    = 3'd1;
    localparam S_STREAM      = 3'd2;
    localparam S_REPLAY_WAIT = 3'd3;
    localparam S_DRAIN       = 3'd4;
    localparam S_DONE        = 3'd6;


    // pipe_run gates the local input-shift pipelines. When it is low, the
    // diagonal aligners and array-facing output registers hold their values.
    wire pipe_run = ((state == S_WGT_LEAD) && wgt_stream_en) ||
                    ((state == S_STREAM) &&
                     act_stream_en &&
                     (wgt_stream_en || wgt_all_read_done)) ||
                    (state == S_DRAIN);

    always@(posedge i_clk or negedge i_rst_n)begin
        if(!i_rst_n) begin
            act_out_en_r <=0;
            wgt_done_r <= 0;
        end
        else if(pipe_run) begin
            act_out_en_r <= act_out_en;
            wgt_done_r <= wgt_done;
        end
    end

    always@(posedge i_clk or negedge i_rst_n)begin
        if(!i_rst_n)
            wgt_done_cnt <= 3'b0;
        else if (wgt_done_cnt == 3'd4)
            wgt_done_cnt <= 3'b0;
        else if(pipe_run && ((wgt_done && !wgt_done_r) || wgt_done_cnt > 3'b0))
            wgt_done_cnt <= wgt_done_cnt + 1;
    end

    assign sel_trigger = (wgt_done_cnt == 3'd3) ? 1 : 0;

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
        else if(!pipe_run )begin
        end
        else if((state == S_DRAIN) && !act_out_en)begin
            o_act_array_data <= 'b0;
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
        else if (wgt_group_cnt_end) begin
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
                if(o_array_en && wgt_out_en)
                    o_wgt_array_ld <= {o_wgt_array_ld[2:0], o_wgt_array_ld[3]};
        end
    end

    // ------------------------------------------------------------------------simple control frame
    // Normal compute cycles: both activation and weight aligners have data.
    wire array_feed_en = act_out_en & wgt_out_en && pipe_run;
    wire array_drain_en = (state == S_DRAIN);

    // Psum scheduling is intentionally removed. The replacement controller
    // must drive the existing psum ports directly.
    assign o_array_en = array_feed_en || array_drain_en;

    // C packing is not implemented in this module.
    assign o_c_data_flat = {MAX_M_SIZE*MAX_N_SIZE*ACC_W{1'b0}};

    // Current activation pass is complete when the last K word of the last
    // M-row block is accepted for this N tile.
    assign act_done = (state == S_STREAM) &&
                      act_stream_en &&
                      (act_k_cnt == (k_size_r - 1'b1)) &&
                      (act_m_cnt == (((m_size_r + 3) >> 2) - 1'b1));

    // Current weight replay is complete when the last K word is accepted.
    assign wgt_done = (state == S_STREAM) &&
                      wgt_stream_en &&
                      (wgt_k_cnt == (k_size_r - 1'b1));

    // ------------------------------------------------------------------------
    // 1/3: state and bookkeeping registers
    // ------------------------------------------------------------------------

    function [3:0] psum_zero_q;
        input [3:0] psum_zero_d;
        begin
            case(psum_zero_d)
                4'b0001: psum_zero_q = 4'b0011;
                4'b0011: psum_zero_q = 4'b0111;
                4'b0111: psum_zero_q = 4'b1111;
                4'b1111: psum_zero_q = 4'b1110;
                4'b1110: psum_zero_q = 4'b1100;
                4'b1100: psum_zero_q = 4'b1000;
                default: psum_zero_q = 4'b0000;
            endcase
        end
    endfunction


    function [3:0] psum_valid_q;
        input [3:0] psum_valid_d;
        input [4:0] mode;
        begin
            psum_valid_q = 4'b0000;
            case (mode)
                5'd1: case (psum_valid_d)
                    4'b0001: psum_valid_q = 4'b0011;
                    4'b0011: psum_valid_q = 4'b0111;
                    4'b0111: psum_valid_q = 4'b1111;
                    4'b1111: psum_valid_q = 4'b1110;
                    4'b1110: psum_valid_q = 4'b1100;
                    4'b1100: psum_valid_q = 4'b1000;
                endcase
                5'd2: case (psum_valid_d)
                    4'b0001: psum_valid_q = 4'b0010;
                    4'b0010: psum_valid_q = 4'b0100;
                    4'b0100: psum_valid_q = 4'b1000;
                endcase
                5'd3: case (psum_valid_d)
                    4'b0001: psum_valid_q = 4'b0011;
                    4'b0011: psum_valid_q = 4'b0110;
                    4'b0110: psum_valid_q = 4'b1100;
                    4'b1100: psum_valid_q = 4'b1000;
                endcase
                5'd4: case (psum_valid_d)
                    4'b0001: psum_valid_q = 4'b0011;
                    4'b0011: psum_valid_q = 4'b0111;
                    4'b0111: psum_valid_q = 4'b1110;
                    4'b1110: psum_valid_q = 4'b1100;
                    4'b1100: psum_valid_q = 4'b1000;
                endcase
                5'd5: psum_valid_q = (psum_valid_d==4'b0000) ? 4'b0000 : (psum_valid_cnt < 2'b11) ? 4'b0001 : 4'b0000;
                5'd6: case (psum_valid_d)
                    4'b0001: psum_valid_q = 4'b0011;
                    4'b0011: psum_valid_q = (psum_valid_d==4'b0000) ? 4'b0000 : (psum_valid_cnt < 2'b11) ? 4'b0011 : 4'b0010;
                endcase
                5'd7: case (psum_valid_d)
                    4'b0001: psum_valid_q = 4'b0011;
                    4'b0011: psum_valid_q = 4'b0111;
                    4'b0111: psum_valid_q = (psum_valid_d==4'b0000) ? 4'b0000 : (psum_valid_cnt < 2'b11) ? 4'b0111 : 4'b0110;
                    4'b0110: psum_valid_q = 4'b0100;
                endcase
                5'd9: psum_valid_q = (psum_valid_d==4'b0000) ? 4'b0000 : (psum_valid_cnt < 2'b01) ? 4'b0001 : 4'b0000;
                5'd10: psum_valid_q = (psum_valid_d==4'b0000) ? 4'b0000 : (psum_valid_cnt < 2'b10) ? 4'b0001 : 4'b0000;
                5'd11: case (psum_valid_d)
                    4'b0001: psum_valid_q = 4'b0010;
                endcase
                5'd12: case (psum_valid_d)
                    4'b0001: psum_valid_q = 4'b0011;
                    4'b0011: psum_valid_q = 4'b0010;
                endcase
                5'd13: case (psum_valid_d)
                    4'b0001: psum_valid_q = 4'b0011;
                    4'b0011: psum_valid_q = (psum_valid_d==4'b0000) ? 4'b0000 : (psum_valid_cnt < 2'b10) ? 4'b0011 : 4'b0010;
                endcase
                5'd14: case (psum_valid_d)
                    4'b0001: psum_valid_q = 4'b0010;
                    4'b0010: psum_valid_q = 4'b0100;
                endcase
                5'd15: case (psum_valid_d)
                    4'b0001: psum_valid_q = 4'b0011;
                    4'b0011: psum_valid_q = 4'b0110;
                    4'b0110: psum_valid_q = 4'b0100;
                endcase
                5'd16: case (psum_valid_d)
                    4'b0001: psum_valid_q = 4'b0011;
                    4'b0011: psum_valid_q = 4'b0111;
                    4'b0111: psum_valid_q = 4'b0110;
                    4'b0110: psum_valid_q = 4'b0100;
                endcase
            endcase
        end
    endfunction



    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state               <= S_IDLE;
            act_base_req_q      <= 1'b0;
            wgt_base_req_q      <= 1'b0;
            o_act_base_addr     <= 32'b0;
            o_wgt_base_addr     <= 32'b0;
            o_psum_sel          <= 8'b0;
            m_size_r            <= 16'b0;
            n_size_r            <= 16'b0;
            k_size_r            <= 16'b0;
            act_k_cnt           <= 16'b0;
            wgt_k_cnt           <= 16'b0;
            act_m_cnt           <= 16'b0;
            act_group_cnt       <= 14'b0;
            wgt_group_cnt       <= 14'b0;
            act_all_read_done   <= 1'b0;
            wgt_all_read_done   <= 1'b0;
            psum_valid[0]       <= 4'b0;
            psum_valid[1]       <= 4'b0;
            psum_valid_cnt      <= 2'b0;
            psum_valid_mode[0]  <= 5'b0;
            psum_valid_mode[1]  <= 5'b0;
            mode_switch         <= 1'b0;
            o_array_clear <= 1'b1;
            drain_cnt <= 2'b0;
        end
        else begin
            state            <= state_d;
            act_base_req_q <= 1'b0;
            wgt_base_req_q <= 1'b0;

            if ((state == S_IDLE) && i_size_ld) begin
                m_size_r          <= i_m_size;
                n_size_r          <= i_n_size;
                k_size_r          <= i_k_size;
                act_k_cnt         <= 16'b0;
                wgt_k_cnt         <= 16'b0;
                act_m_cnt         <= 16'b0;
                act_group_cnt     <= 14'b0;
                wgt_group_cnt     <= 14'b0;
                act_all_read_done <= 1'b0;
                wgt_all_read_done <= 1'b0;
                psum_valid[0]     <= 4'b0;
                psum_valid[1]     <= 4'b0;
                psum_valid_cnt    <= 2'b0;
                psum_valid_mode[0] <= 5'b0;
                psum_valid_mode[1] <= 5'b0;
                mode_switch       <= 1'b0;
                o_act_base_addr   <= i_act_base_addr;
                o_wgt_base_addr   <= i_wgt_base_addr;
                act_base_req_q    <= 1'b1;
                wgt_base_req_q    <= 1'b1;
                o_array_clear <= 1'b1;
                drain_cnt <= 2'b0;
            end

            if ((state == S_WGT_LEAD) && wgt_stream_en) begin
                o_array_clear <= 1'b0;
                o_psum_sel  <= 8'hff;
                if (wgt_k_cnt == (k_size_r - 1'b1))
                    wgt_k_cnt <= 16'b0;
                else
                    wgt_k_cnt <= wgt_k_cnt + 1'b1;
            end

            if(pipe_run) begin
                // o_psum_sel update
                o_psum_sel[3:2] <= o_psum_sel[1:0];
                o_psum_sel[5:4] <= o_psum_sel[3:2];
                o_psum_sel[7:6] <= o_psum_sel[5:4];
                if(sel_trigger) 
                    o_psum_sel[1:0] <= o_psum_sel[1:0] + k_size_r[1:0];

                psum_valid[0] <= psum_valid_q(psum_valid[0], psum_valid_mode[0]);
                psum_valid[1] <= psum_valid_q(psum_valid[1], psum_valid_mode[1]);

                //o_psum_valid update
                if(sel_trigger) begin
                    mode_switch <= ~mode_switch;
                    psum_valid[mode_switch] <= 4'b0001;
                    psum_valid_cnt  <= 2'b00;
                    if(!wgt_all_read_done && act_group_cnt!= (((n_size_r + 3) >> 2)-1 ))
                        psum_valid_mode[mode_switch] <= 5'd1;
                    else if(wgt_all_read_done && act_group_cnt!= (((n_size_r + 3) >> 2)-1 ))
                        psum_valid_mode[mode_switch] <= (m_size_r[1:0] == 2'b00) ? 5'd1 :
                                                        (m_size_r[1:0] == 2'b01) ? 5'd2 :
                                                        (m_size_r[1:0] == 2'b10) ? 5'd3 : 5'd4 ;
                    else if(!wgt_all_read_done && act_group_cnt== (((n_size_r + 3) >> 2)-1 ))
                        psum_valid_mode[mode_switch] <= (n_size_r[1:0] == 2'b00) ? 5'd1 :
                                                        (n_size_r[1:0] == 2'b01) ? 5'd5 :
                                                        (n_size_r[1:0] == 2'b10) ? 5'd6 : 5'd7 ;
                    else begin
                        case ({n_size_r[1:0], m_size_r[1:0]})
                        4'b0000: psum_valid_mode[mode_switch] <= 5'd1;
                        4'b0001: psum_valid_mode[mode_switch] <= 5'd2;
                        4'b0010: psum_valid_mode[mode_switch] <= 5'd3;
                        4'b0011: psum_valid_mode[mode_switch] <= 5'd4;
                        4'b0100: psum_valid_mode[mode_switch] <= 5'd5;
                        4'b1000: psum_valid_mode[mode_switch] <= 5'd6;
                        4'b1100: psum_valid_mode[mode_switch] <= 5'd7;

                        4'b0101: psum_valid_mode[mode_switch] <= 5'd8;
                        4'b0110: psum_valid_mode[mode_switch] <= 5'd9;
                        4'b0111: psum_valid_mode[mode_switch] <= 5'd10;
                        4'b1001: psum_valid_mode[mode_switch] <= 5'd11;
                        4'b1010: psum_valid_mode[mode_switch] <= 5'd12;
                        4'b1011: psum_valid_mode[mode_switch] <= 5'd13;
                        4'b1101: psum_valid_mode[mode_switch] <= 5'd14;
                        4'b1110: psum_valid_mode[mode_switch] <= 5'd15;
                        4'b1111: psum_valid_mode[mode_switch] <= 5'd16;
                        endcase
                    end
                end

                if(psum_valid_cnt < 2'b11)begin
                    psum_valid_cnt  <= psum_valid_cnt + 2'b01;
                end

            end

            if (state == S_STREAM) begin
                o_array_clear <= 1'b0;

                if (wgt_stream_en) begin
                    if (wgt_k_cnt == (k_size_r - 1'b1))
                        wgt_k_cnt <= 16'b0;
                    else
                        wgt_k_cnt <= wgt_k_cnt + 1'b1;
                end

                if (wgt_done) begin
                    if (wgt_group_cnt == ((m_size_r + 3) >> 2)-1'b1) begin
                        wgt_group_cnt    <= wgt_group_cnt + 1'b1;
                        wgt_all_read_done <= 1'b1;
                    end
                    else begin
                        wgt_group_cnt    <= wgt_group_cnt + 1'b1;
                        wgt_base_req_q <= 1'b1;
                    end
                end

                if (act_done) begin
                    act_k_cnt <= 16'b0;
                    wgt_k_cnt <= 16'b0;
                    act_m_cnt <= 16'b0;
                end
                else if (act_stream_en) begin
                    if (act_k_cnt == (k_size_r - 1'b1)) begin
                        act_k_cnt <= 16'b0;
                        act_m_cnt <= act_m_cnt + 1'b1;
                    end
                    else begin
                        act_k_cnt <= act_k_cnt + 1'b1;
                    end
                end
            end


            if ((state == S_DRAIN) && !act_out_en ) begin
                if(drain_cnt < 2'b11)
                    drain_cnt <= drain_cnt + 2'b01;
                else
                    drain_cnt <= 2'b0;
                

                if(act_out_en_r && !act_out_en)begin
                    if (act_group_cnt == (((n_size_r + 3) >> 2)-1 )) begin
                        act_all_read_done <= 1'b1;
                        act_group_cnt     <= act_group_cnt + 1'b1;
                        wgt_group_cnt     <= 14'b0;
                        wgt_k_cnt         <= 16'b0;
                        wgt_all_read_done <= 1'b0;
                        o_act_base_addr   <= i_act_base_addr;
                        o_wgt_base_addr   <= o_wgt_base_addr + k_size_r;
                        act_base_req_q    <= 1'b1;
                        wgt_base_req_q    <= 1'b1;
                    end
                    else begin
                        act_group_cnt     <= act_group_cnt + 1'b1;
                        wgt_group_cnt     <= 14'b0;
                        wgt_k_cnt         <= 16'b0;
                        wgt_all_read_done <= 1'b0;
                        o_act_base_addr   <= i_act_base_addr;
                        o_wgt_base_addr   <= o_wgt_base_addr + k_size_r;
                        act_base_req_q    <= 1'b1;
                        wgt_base_req_q    <= 1'b1;
                    end
                end
            end

            if((state == S_STREAM || state ==S_DRAIN))begin
                // if((wgt_stream_en & act_stream_en)|| state ==S_DRAIN)begin
                if(o_array_en)begin
                    if(sel_trigger && (o_psum_sel[1:0] + k_size_r[1:0] == 2'b11)) 
                        o_psum_zero[3:0] <= 4'b0001;
                    else 
                        o_psum_zero[3:0] <= psum_zero_q(o_psum_zero[3:0]);

                    if(sel_trigger && (o_psum_sel[1:0] + k_size_r[1:0] == 2'b00)) 
                        o_psum_zero[7:4] <= 4'b0001;
                    else
                        o_psum_zero[7:4] <= psum_zero_q(o_psum_zero[7:4]);

                    if(sel_trigger && (o_psum_sel[1:0] + k_size_r[1:0] == 2'b01)) 
                        o_psum_zero[11:8] <= 4'b0001;
                    else
                        o_psum_zero[11:8] <= psum_zero_q(o_psum_zero[11:8]);  

                    if(sel_trigger && (o_psum_sel[1:0] + k_size_r[1:0] == 2'b10)) 
                        o_psum_zero[15:12] <= 4'b0001;
                    else
                        o_psum_zero[15:12] <= psum_zero_q(o_psum_zero[15:12]);                  
                end
            end
        end
    end

    // ------------------------------------------------------------------------
    // 2/3: next-state decode
    // ------------------------------------------------------------------------
    always @* begin
        state_d = state;

        case (state)
            S_IDLE:
                if (i_size_ld)
                    state_d = S_WGT_LEAD;

            S_WGT_LEAD:
                if (wgt_stream_en) begin
                    if (i_act_stream_valid)
                        state_d = S_STREAM;
                    else
                        state_d = S_REPLAY_WAIT;
                end

            S_STREAM:
                if (act_done)
                    state_d = S_DRAIN;
                else if (wgt_done &&
                         (wgt_group_cnt != (((m_size_r + 3) >> 2) - 1'b1)))
                    state_d = S_REPLAY_WAIT;

            S_REPLAY_WAIT:
                if (i_act_stream_valid && i_wgt_stream_valid)
                    state_d = S_STREAM;

            S_DRAIN:
                if (drain_cnt == 2'b11) begin
                    if (act_group_cnt == (((n_size_r + 3) >> 2)))
                        state_d = S_DONE;
                    else
                        state_d = S_WGT_LEAD;
                end

            S_DONE:
                state_d = S_IDLE;

            default:
                state_d = S_IDLE;
        endcase
    end

    // ------------------------------------------------------------------------
    // 3/3: output/control decode
    // ------------------------------------------------------------------------
    always @* begin
        o_act_stream_ready = 1'b0;
        o_wgt_stream_ready = 1'b0;
        o_act_base_valid   = act_base_req_q;
        o_wgt_base_valid   = wgt_base_req_q;

        case (state)
            S_WGT_LEAD:
                o_wgt_stream_ready = 1'b1;

            S_STREAM: begin
                if (wgt_all_read_done) begin
                    o_act_stream_ready = 1'b1;
                end
                else begin
                    o_act_stream_ready = i_wgt_stream_valid;
                    o_wgt_stream_ready = i_act_stream_valid;
                end
            end

            default: begin
            end
        endcase
    end

endmodule
