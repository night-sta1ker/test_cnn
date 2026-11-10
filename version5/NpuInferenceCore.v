`timescale 1ns / 1ps

module NpuInferenceCore (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output wire         busy,
    output wire         done,
    output wire [3:0]   class_index,
    output wire [7:0]   class_score,
    output reg          error,

    input  wire         host_act_wr_en,
    input  wire [9:0]   host_act_wr_addr,
    input  wire [31:0]  host_act_wr_data,
    input  wire [3:0]   host_act_wr_be,

    input  wire         model_wgt_wr_en,
    input  wire [9:0]   model_wgt_wr_addr,
    input  wire [31:0]  model_wgt_wr_data,
    input  wire         model_param_wr_en,
    input  wire [4:0]   model_param_wr_addr,
    input  wire [64:0]  model_param_wr_data
);
    wire gemm_start;
    wire pool_start;
    wire argmax_start;
    wire [2:0] stage;
    wire act_mode;
    wire [15:0] act_h;
    wire [15:0] act_w;
    wire [2:0] act_channels;
    wire [15:0] act_channel_stride;
    wire [15:0] act_vector_size;
    wire [11:0] act_base_byte;
    wire [15:0] seq_m;
    wire [15:0] seq_k;
    wire [15:0] seq_n;
    wire [9:0] wgt_base_word;
    wire [11:0] result_base_byte;
    wire [5:0] result_out_w;
    wire [5:0] result_row_stride;
    wire [11:0] result_plane_size;
    wire [4:0] param_base;
    wire [11:0] pool_src_base_byte;
    wire [11:0] pool_dst_base_byte;
    wire [3:0] pool_channels;
    wire [5:0] pool_h;
    wire [5:0] pool_w;
    wire [11:0] pool_plane_stride;
    wire [11:0] argmax_src_base_byte;

    wire act_cfg_ready;
    wire act_rd_en;
    wire [9:0] act_rd_addr;
    wire [31:0] act_stream_data;
    wire act_stream_valid;
    wire act_stream_ready;

    wire [31:0] wgt_stream_data;
    wire wgt_stream_valid;
    wire wgt_stream_ready;
    wire [31:0] seq_act_base_addr;
    wire seq_act_base_valid;
    wire [31:0] seq_wgt_base_addr;
    wire seq_wgt_base_valid;

    wire array_en;
    wire array_clear;
    wire [31:0] array_act_data;
    wire [3:0] array_wgt_ld;
    wire [31:0] array_wgt_data;
    wire [15:0] array_psum_zero;
    wire [7:0] array_psum_sel;
    wire [127:0] array_psum_data;
    wire [3:0] seq_psum_valid;
    wire [15:0] result_m0;
    wire [15:0] result_m1;
    wire [15:0] result_m2;
    wire [15:0] result_m3;
    wire [15:0] result_n_base;
    wire seq_done;
    wire post_psum_ready;

    wire [4:0] param_rd_base = param_base + result_n_base[4:0];
    wire [127:0] param_bias;
    wire [63:0] param_mult;
    wire [31:0] param_shift;
    wire [31:0] param_zout;
    wire [3:0] param_relu;

    wire post_done;
    wire post_overflow;
    wire post_protocol_error;
    wire post_wr_en;
    wire [9:0] post_wr_addr;
    wire [31:0] post_wr_data;
    wire [3:0] post_wr_be;

    wire pool_done;
    wire pool_rd_en;
    wire [9:0] pool_rd_addr;
    wire pool_wr_en;
    wire [9:0] pool_wr_addr;
    wire [31:0] pool_wr_data;
    wire [3:0] pool_wr_be;

    wire argmax_done;
    wire argmax_rd_en;
    wire [9:0] argmax_rd_addr;

    wire sram_rd_en = (stage == 3'd2 || stage == 3'd4) ? pool_rd_en :
                      (stage == 3'd6) ? argmax_rd_en : act_rd_en;
    wire [9:0] sram_rd_addr = (stage == 3'd2 || stage == 3'd4) ? pool_rd_addr :
                              (stage == 3'd6) ? argmax_rd_addr : act_rd_addr;
    wire [31:0] sram_rd_data;
    wire compute_wr_en = (stage == 3'd2 || stage == 3'd4) ? pool_wr_en : post_wr_en;
    wire [9:0] compute_wr_addr = (stage == 3'd2 || stage == 3'd4) ?
                                 pool_wr_addr : post_wr_addr;
    wire [31:0] compute_wr_data = (stage == 3'd2 || stage == 3'd4) ?
                                  pool_wr_data : post_wr_data;
    wire [3:0] compute_wr_be = (stage == 3'd2 || stage == 3'd4) ?
                               pool_wr_be : post_wr_be;
    wire load_wr_allowed = !busy;
    wire sram_wr_en = busy ? compute_wr_en : host_act_wr_en;
    wire [9:0] sram_wr_addr = busy ? compute_wr_addr : host_act_wr_addr;
    wire [31:0] sram_wr_data = busy ? compute_wr_data : host_act_wr_data;
    wire [3:0] sram_wr_be = busy ? compute_wr_be : host_act_wr_be;

    wire protocol_fault = post_overflow || post_protocol_error ||
                          (busy && (start || host_act_wr_en ||
                                    model_wgt_wr_en || model_param_wr_en)) ||
                          (gemm_start && !act_cfg_ready);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            error <= 1'b0;
        else if (start && !busy)
            error <= 1'b0;
        else if (protocol_fault)
            error <= 1'b1;
    end

    LayerController controller_u (
        .clk(clk), .rst_n(rst_n), .start(start),
        .gemm_done(post_done), .pool_done(pool_done), .argmax_done(argmax_done),
        .busy(busy), .done(done), .gemm_start(gemm_start),
        .pool_start(pool_start), .argmax_start(argmax_start), .stage(stage),
        .act_mode(act_mode), .act_h(act_h), .act_w(act_w),
        .act_channels(act_channels), .act_channel_stride(act_channel_stride),
        .act_vector_size(act_vector_size), .act_base_byte(act_base_byte),
        .seq_m(seq_m), .seq_k(seq_k), .seq_n(seq_n),
        .wgt_base_word(wgt_base_word), .result_base_byte(result_base_byte),
        .result_out_w(result_out_w), .result_row_stride(result_row_stride),
        .result_plane_size(result_plane_size), .param_base(param_base),
        .pool_src_base_byte(pool_src_base_byte),
        .pool_dst_base_byte(pool_dst_base_byte), .pool_channels(pool_channels),
        .pool_h(pool_h), .pool_w(pool_w), .pool_plane_stride(pool_plane_stride),
        .argmax_src_base_byte(argmax_src_base_byte)
    );

    ActivationSram activation_sram_u (
        .clk(clk), .rd_en(sram_rd_en), .rd_addr(sram_rd_addr),
        .rd_data(sram_rd_data), .wr_en(sram_wr_en), .wr_addr(sram_wr_addr),
        .wr_data(sram_wr_data), .wr_be(sram_wr_be)
    );

    ActBuffer act_buffer_u (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .cfg_valid(gemm_start), .cfg_ready(act_cfg_ready), .cfg_mode(act_mode),
        .cfg_h(act_h), .cfg_w(act_w), .cfg_channels(act_channels),
        .cfg_channel_stride(act_channel_stride), .cfg_vector_size(act_vector_size),
        .req_valid(seq_act_base_valid), .req_base_byte(seq_act_base_addr[11:0]),
        .sram_rd_en(act_rd_en), .sram_rd_addr(act_rd_addr),
        .sram_rd_data(sram_rd_data), .seq_data(act_stream_data),
        .seq_valid(act_stream_valid), .seq_ready(act_stream_ready)
    );

    WgtBuffer wgt_buffer_u (
        .clk(clk), .rst_n(rst_n), .clear(gemm_start),
        .wr_en(model_wgt_wr_en && load_wr_allowed), .wr_addr(model_wgt_wr_addr),
        .wr_data(model_wgt_wr_data), .base_valid(seq_wgt_base_valid),
        .base_addr(seq_wgt_base_addr[9:0]), .k_size(seq_k),
        .stream_data(wgt_stream_data), .stream_valid(wgt_stream_valid),
        .stream_ready(wgt_stream_ready)
    );

    ParamBuffer param_buffer_u (
        .clk(clk), .wr_en(model_param_wr_en && load_wr_allowed),
        .wr_addr(model_param_wr_addr),
        .wr_data(model_param_wr_data), .rd_base(param_rd_base),
        .rd_bias(param_bias), .rd_mult(param_mult), .rd_shift(param_shift),
        .rd_zout(param_zout), .rd_relu(param_relu)
    );

    Sequencer sequencer_u (
        .i_clk(clk), .i_rst_n(rst_n), .i_size_ld(gemm_start),
        .i_m_size(seq_m), .i_k_size(seq_k), .i_n_size(seq_n),
        .i_act_base_addr({20'b0, act_base_byte}),
        .i_wgt_base_addr({22'b0, wgt_base_word}),
        .i_act_stream_data(act_stream_data),
        .i_act_stream_valid(act_stream_valid),
        .o_act_stream_ready(act_stream_ready),
        .o_act_base_addr(seq_act_base_addr), .o_act_base_valid(seq_act_base_valid),
        .i_wgt_stream_data(wgt_stream_data),
        .i_wgt_stream_valid(wgt_stream_valid),
        .o_wgt_stream_ready(wgt_stream_ready),
        .o_wgt_base_addr(seq_wgt_base_addr), .o_wgt_base_valid(seq_wgt_base_valid),
        .o_array_en(array_en), .o_array_clear(array_clear),
        .o_act_array_data(array_act_data), .o_wgt_array_ld(array_wgt_ld),
        .o_wgt_array_data(array_wgt_data), .o_psum_zero(array_psum_zero),
        .o_psum_sel(array_psum_sel), .i_psum_data(array_psum_data),
        .i_psum_ready(post_psum_ready),
        .o_psum_valid(seq_psum_valid), .o_result_m0(result_m0),
        .o_result_m1(result_m1), .o_result_m2(result_m2),
        .o_result_m3(result_m3), .o_result_n_base(result_n_base),
        .o_done(seq_done)
    );

    SystolicArray4x4 array_u (
        .clk(clk), .rst_n(rst_n), .en(array_en), .wgt_ld(array_wgt_ld),
        .wgt_in(array_wgt_data), .clear(array_clear), .act_in(array_act_data),
        .psum_zero(array_psum_zero), .psum_out_sel(array_psum_sel),
        .psum_out(array_psum_data)
    );

    PostProcessWriter post_writer_u (
        .clk(clk), .rst_n(rst_n), .clear(1'b0), .cfg_valid(gemm_start),
        .cfg_mode_fc(act_mode), .cfg_result_base_byte(result_base_byte),
        .cfg_out_w(result_out_w), .cfg_row_stride(result_row_stride),
        .cfg_plane_size(result_plane_size), .cfg_m_size(seq_m),
        .cfg_n_size(seq_n), .psum_data(array_psum_data),
        .psum_valid(seq_psum_valid), .psum_ready(post_psum_ready),
        .result_m0(result_m0),
        .result_m1(result_m1), .result_m2(result_m2), .result_m3(result_m3),
        .result_n_base(result_n_base), .seq_done(seq_done),
        .param_bias(param_bias), .param_mult(param_mult),
        .param_shift(param_shift), .param_zout(param_zout),
        .param_relu(param_relu), .busy(), .done(post_done),
        .overflow(post_overflow), .protocol_error(post_protocol_error),
        .sram_wr_en(post_wr_en), .sram_wr_addr(post_wr_addr),
        .sram_wr_data(post_wr_data), .sram_wr_be(post_wr_be)
    );

    MaxPool2x2 pool_u (
        .clk(clk), .rst_n(rst_n), .clear(1'b0), .start(pool_start),
        .src_base_byte(pool_src_base_byte), .dst_base_byte(pool_dst_base_byte),
        .channels(pool_channels), .in_h(pool_h), .in_w(pool_w),
        .src_plane_stride(pool_plane_stride), .busy(), .done(pool_done),
        .sram_rd_en(pool_rd_en), .sram_rd_addr(pool_rd_addr),
        .sram_rd_data(sram_rd_data), .sram_wr_en(pool_wr_en),
        .sram_wr_addr(pool_wr_addr), .sram_wr_data(pool_wr_data),
        .sram_wr_be(pool_wr_be)
    );

    ArgMax10 argmax_u (
        .clk(clk), .rst_n(rst_n), .clear(1'b0), .start(argmax_start),
        .src_base_byte(argmax_src_base_byte), .busy(), .done(argmax_done),
        .class_index(class_index), .max_value(class_score),
        .sram_rd_en(argmax_rd_en), .sram_rd_addr(argmax_rd_addr),
        .sram_rd_data(sram_rd_data)
    );
endmodule
