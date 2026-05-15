`timescale 1ns / 1ps

module Sequencer #(
    parameter DATA_W = 8,
    parameter ACC_W  = 32,
    parameter MAX_M_SIZE = 4,
    parameter MAX_K_SIZE = 4,
    parameter MAX_N_SIZE = 4
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     valid,
    output reg                      ready,

    input  wire                     size_ld,
    input  wire [15:0]              m_size,
    input  wire [15:0]              k_size,
    input  wire [15:0]              n_size,

    input  wire [4*DATA_W-1:0]      act_stream_in,
    input  wire                     act_stream_valid,
    output wire                     act_stream_ready,

    input  wire [4*DATA_W-1:0]      wgt_stream_in,
    input  wire                     wgt_stream_valid,
    output wire                     wgt_stream_ready,
    output wire                     wgt_stream_start,
    output wire [15:0]              wgt_stream_n_base,
    output wire                     act_stream_start,

    output wire [3:0]               wgt_ld,
    output wire [4*DATA_W-1:0]      wgt_in,
    output reg                      en,
    output reg                      clear,
    output wire [4*DATA_W-1:0]      act_in,
    output reg  [15:0]              psum_zero,
    output reg  [1:0]               psum_out_sel,

    input  wire [4*ACC_W-1:0]       psum_out,

    output reg                      done,
    output reg  [31:0]              cycle,
    output wire [MAX_M_SIZE*MAX_N_SIZE*ACC_W-1:0] c_out_flat
);

reg [15:0] m_groups; // ceil(m_size / 4)
reg [15:0] n_groups; // ceil(n_size / 4)
reg [15:0] n_base;   // current N group column base: 0, 4, 8, ...
reg [15:0] wgt_age;  // weight stream age inside current N group
reg [15:0] wgt_replay_idx;
reg  [1:0] group_out_row;// signing this gourp outputs at row 0/1/2/3
reg [3:0]  wgt_phase;

wire [4*DATA_W-1:0]      act_trans_out;
wire                     act_trans_out_valid;
wire                     act_trans_done;
reg                      act_restart;
reg  [4*DATA_W-1:0]      act_hold_data;
reg                      act_hold_valid;

reg                      wgt_block_start;
reg  [1:0]               wgt_restart_mask;
reg                      wgt_lead_active;
reg                      wgt_block_pending;
reg                      wgt_replay_pending;
reg  [1:0]               wgt_lead_count;
wire [4*DATA_W-1:0]      wgt_trans_out;
wire                     wgt_trans_out_valid;
wire                     wgt_trans_block_done;
wire                     wgt_trans_done;
wire                     act_out_ready;
wire                     wgt_out_ready;
wire                     act_src_valid;
wire [4*DATA_W-1:0]      act_src_data;
wire                     act_tail_phase;
wire                     act_pair_allowed;
wire                     act_array_valid;
wire                     wgt_array_valid;
wire [15:0]              wgt_block_n_size;
wire [15:0]              wgt_replay_start_age;
wire [15:0]              wgt_last_out_age;
wire                     wgt_final_replay;

reg [2:0] state;
parameter S_IDLE = 3'b000,
          S_PRE = 3'b001,
          S_COMPUTE_NGROUP = 3'b010,
          S_NX_NGROUP = 3'b011,
          S_DONE = 3'b100;

localparam WGT_LD_INIT = 4'b0001;

assign act_tail_phase = wgt_final_replay &&
                        (wgt_block_pending || wgt_trans_block_done || wgt_trans_done);
assign act_src_valid = act_hold_valid || act_trans_out_valid;
assign act_src_data = act_hold_valid ? act_hold_data : act_trans_out;
assign act_pair_allowed = ((wgt_lead_count != 0) && !wgt_replay_pending) ||
                          wgt_array_valid ||
                          act_tail_phase;
assign act_array_valid = act_src_valid && act_pair_allowed;
assign wgt_array_valid = wgt_trans_out_valid && (wgt_restart_mask == 0);

assign act_in = act_array_valid ? act_src_data : {4*DATA_W{1'b0}};
assign wgt_in = wgt_array_valid ? wgt_trans_out : {4*DATA_W{1'b0}};
assign wgt_ld = wgt_array_valid ? wgt_phase : 4'b0000;
assign wgt_out_ready = (state == S_COMPUTE_NGROUP) &&
                       (act_src_valid ||
                        ((wgt_lead_count == 0) &&
                         !(wgt_array_valid && !act_src_valid)));
assign act_out_ready = (state == S_COMPUTE_NGROUP) &&
                       !act_hold_valid &&
                       act_pair_allowed;
assign wgt_stream_start = wgt_block_start;
assign wgt_stream_n_base = n_base;
assign act_stream_start = act_restart;
assign wgt_block_n_size = ((n_size - n_base) >= 4) ? 16'd4 : (n_size - n_base);
assign wgt_replay_start_age = (k_size > 1) ? (k_size - 2) : 16'd0;
assign wgt_last_out_age = k_size + wgt_block_n_size - 2;
assign wgt_final_replay = ((wgt_replay_idx + 1'b1) >= m_groups);

trans_act #(
    .DATA_W(DATA_W),
    .MAX_M_SIZE(MAX_M_SIZE),
    .MAX_K_SIZE(MAX_K_SIZE)
) trans_act_u (
    .clk            (clk),
    .rst_n          (rst_n),
    .restart        (act_restart),
    .m_size         (m_size),
    .k_size         (k_size),
    .in_data        (act_stream_in),
    .in_data_valid  (act_stream_valid),
    .in_data_ready  (act_stream_ready),
    .out_data       (act_trans_out),
    .out_data_valid (act_trans_out_valid),
    .out_data_ready (act_out_ready),
    .done           (act_trans_done)
);

trans_wgt #(
    .DATA_W(DATA_W)
) trans_wgt_u (
    .clk            (clk),
    .rst_n          (rst_n),
    .valid          (wgt_block_start),
    .k_size         (k_size),
    .n_size         (wgt_block_n_size),
    .in_data        (wgt_stream_in),
    .in_data_valid  (wgt_stream_valid),
    .in_data_ready  (wgt_stream_ready),
    .out_data       (wgt_trans_out),
    .out_data_valid (wgt_trans_out_valid),
    .out_data_ready (wgt_out_ready),
    .block_done     (wgt_trans_block_done),
    .done           (wgt_trans_done)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= S_IDLE;
        ready <= 1'b1;
        wgt_phase <= WGT_LD_INIT;
        en <= 0;
        clear <= 0;
        psum_zero <= 0;
        psum_out_sel <= 0;
        done <= 0;
        cycle <= 0;
        m_groups <= 0;
        n_groups <= 0;
        n_base <= 0;
        wgt_age <= 0;
        wgt_replay_idx <= 0;
        wgt_block_start <= 1'b0;
        wgt_restart_mask <= 0;
        act_restart <= 1'b0;
        act_hold_data <= 0;
        act_hold_valid <= 1'b0;
        wgt_lead_active <= 1'b0;
        wgt_block_pending <= 1'b0;
        wgt_replay_pending <= 1'b0;
        wgt_lead_count <= 0;
    end else begin
        wgt_block_start <= 1'b0;
        if (wgt_restart_mask != 0)
            wgt_restart_mask <= wgt_restart_mask - 1'b1;
        act_restart <= 1'b0;

        case (state)
            S_IDLE: begin
                ready <= 1'b1;
                en <= 1'b0;
                clear <= 1'b0;
                done <= 1'b0;
                cycle <= 0;

                if (valid && ready) begin
                    ready <= 1'b0;
                    state <= S_PRE;
                end
            end

            S_PRE: begin
                ready <= 1'b0;
                en <= 1'b0;
                clear <= 1'b0;
                m_groups <= (m_size + 3) >> 2;
                n_groups <= (n_size + 3) >> 2;
                n_base <= 0;
                wgt_age <= 0;
                wgt_replay_idx <= 0;
                wgt_phase <= WGT_LD_INIT;
                wgt_block_start <= 1'b1;
                wgt_restart_mask <= 2;
                act_restart <= 1'b1;
                act_hold_data <= 0;
                act_hold_valid <= 1'b0;
                wgt_lead_active <= 1'b0;
                wgt_block_pending <= 1'b0;
                wgt_replay_pending <= 1'b0;
                wgt_lead_count <= 0;
                state <= S_COMPUTE_NGROUP;
            end

            S_COMPUTE_NGROUP: begin
                ready <= 1'b0;
                en <= 1'b1;
                clear <= 1'b0;
                cycle <= cycle + 1;

                if (wgt_array_valid) begin
                    wgt_lead_active <= 1'b1;
                    wgt_phase <= {wgt_phase[2:0], wgt_phase[3]};
                    wgt_age <= wgt_age + 1;
                end

                if (act_hold_valid && act_array_valid) begin
                    act_hold_valid <= 1'b0;
                end else if (!act_hold_valid && act_trans_out_valid &&
                             !act_pair_allowed) begin
                    act_hold_data <= act_trans_out;
                    act_hold_valid <= 1'b1;
                end

                if (wgt_array_valid && !act_array_valid) begin
                    if (wgt_lead_count != 2'd3)
                        wgt_lead_count <= wgt_lead_count + 1'b1;
                end else if (!wgt_array_valid && act_array_valid) begin
                    if (wgt_lead_count != 0)
                        wgt_lead_count <= wgt_lead_count - 1'b1;
                end

                if (wgt_trans_block_done && wgt_array_valid)
                    wgt_block_pending <= 1'b1;

                if (!wgt_final_replay && !wgt_replay_pending &&
                    (wgt_age >= wgt_replay_start_age)) begin
                    wgt_block_start <= 1'b1;
                    wgt_restart_mask <= 2;
                    wgt_replay_pending <= 1'b1;
                    wgt_age <= 0;
                    wgt_phase <= WGT_LD_INIT;
                    wgt_lead_active <= 1'b0;
                    wgt_lead_count <= 0;
                end

                if (wgt_array_valid && wgt_replay_pending &&
                    (wgt_age >= wgt_last_out_age)) begin
                    wgt_replay_pending <= 1'b0;
                    wgt_replay_idx <= wgt_replay_idx + 1'b1;
                    wgt_age <= 0;
                    wgt_phase <= WGT_LD_INIT;
                    wgt_lead_active <= 1'b0;
                    wgt_lead_count <= 0;
                end

                if (wgt_final_replay && act_trans_done) begin
                    if ((n_base + 4) >= n_size)
                        state <= S_DONE;
                    else
                        state <= S_NX_NGROUP;
                end else begin
                    state <= S_COMPUTE_NGROUP;
                end
            end

            S_NX_NGROUP: begin
                ready <= 1'b0;
                en <= 1'b1;
                clear <= 1'b1;
                n_base <= n_base + 4;
                wgt_age <= 0;
                wgt_replay_idx <= 0;
                wgt_phase <= WGT_LD_INIT;
                wgt_block_start <= 1'b1;
                wgt_restart_mask <= 2;
                act_restart <= 1'b1;
                act_hold_data <= 0;
                act_hold_valid <= 1'b0;
                wgt_lead_active <= 1'b0;
                wgt_block_pending <= 1'b0;
                wgt_replay_pending <= 1'b0;
                wgt_lead_count <= 0;
                state <= S_COMPUTE_NGROUP;
            end

            S_DONE: begin
                ready <= 1'b1;
                en <= 1'b0;
                clear <= 1'b0;
                done <= 1'b1;
            end

            default: begin
                state <= S_IDLE;
                ready <= 1'b1;
                en <= 1'b0;
                clear <= 1'b0;
            end
        endcase
    end
end




endmodule
