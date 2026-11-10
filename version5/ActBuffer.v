`timescale 1ns / 1ps

module ActBuffer #(
    parameter SRAM_ADDR_W = 10,
    parameter FIFO_ADDR_W = 7
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clear,

    input  wire                    cfg_valid,
    output wire                    cfg_ready,
    input  wire                    cfg_mode,       // 0: CONV3X3, 1: FC
    input  wire [15:0]             cfg_h,
    input  wire [15:0]             cfg_w,
    input  wire [2:0]              cfg_channels,
    input  wire [15:0]             cfg_channel_stride,
    input  wire [15:0]             cfg_vector_size,

    // Sequencer emits a one-cycle request and has no request-ready input.
    input  wire                    req_valid,
    input  wire [11:0]             req_base_byte,

    output wire                    sram_rd_en,
    output wire [SRAM_ADDR_W-1:0]  sram_rd_addr,
    input  wire [31:0]             sram_rd_data,

    output wire [31:0]             seq_data,
    output wire                    seq_valid,
    input  wire                    seq_ready
);

    localparam MODE_CONV = 1'b0;
    localparam MODE_FC   = 1'b1;

    localparam L_IDLE  = 3'd0;
    localparam L_CHECK = 3'd1;
    localparam L_REQ   = 3'd2;
    localparam L_WAIT  = 3'd3;
    localparam L_PIXEL = 3'd4;

    localparam G_IDLE  = 3'd0;
    localparam G_RD0   = 3'd1;
    localparam G_RD1   = 3'd2;
    localparam G_CAP1  = 3'd3;
    localparam G_EMIT  = 3'd4;

    localparam FIFO_DEPTH = (1 << FIFO_ADDR_W);

    reg        mode_r;
    reg [15:0] h_r;
    reg [15:0] w_r;
    reg [2:0]  channels_r;
    reg [15:0] channel_stride_r;
    reg [15:0] vector_size_r;
    reg [15:0] out_w_r;
    reg [15:0] out_h_r;
    reg [15:0] groups_r;

    reg        req_pending;
    reg [11:0] req_base_r;
    reg        active;

    reg [2:0]  load_state;
    reg [15:0] load_row;
    reg [2:0]  load_ch;
    reg [15:0] load_col;
    reg [11:0] load_row_base;
    reg [11:0] load_ch_base;
    reg [11:0] load_byte_addr;
    reg [31:0] load_pack;
    reg [15:0] fc_index;

    reg        cache_valid;
    reg [SRAM_ADDR_W-1:0] cache_tag;
    reg [31:0] cache_data;

    reg [3:0]  row_valid;
    reg [15:0] row_tag [0:3];

    reg [2:0]  gen_state;
    reg [15:0] gen_y;
    reg [15:0] gen_group;
    reg [2:0]  gen_ch;
    reg [1:0]  gen_kh;
    reg [1:0]  gen_kw;
    reg [31:0] gen_word0;
    reg [31:0] gen_word1;

    reg [FIFO_ADDR_W-1:0] fifo_wr_ptr;
    reg [FIFO_ADDR_W-1:0] fifo_rd_ptr;
    reg [FIFO_ADDR_W:0]   fifo_count;
    wire [31:0]           fifo_rd_data;
    reg [31:0]            fifo_out_data;
    reg                   fifo_out_valid;
    reg [31:0]            fifo_next_data;
    reg                   fifo_next_valid;
    reg                   fifo_rd_pending;

    wire fifo_empty = (fifo_count == 0);
    wire fifo_full = (fifo_count == FIFO_DEPTH);
    wire fifo_pop = fifo_out_valid && seq_ready;
    wire [FIFO_ADDR_W:0] fifo_front_count =
        {{FIFO_ADDR_W{1'b0}}, fifo_out_valid} +
        {{FIFO_ADDR_W{1'b0}}, fifo_next_valid} +
        {{FIFO_ADDR_W{1'b0}}, fifo_rd_pending};
    wire fifo_mem_has_data = (fifo_count > fifo_front_count);
    wire fifo_read_issue = fifo_mem_has_data && !fifo_rd_pending &&
                           (!fifo_next_valid || fifo_pop);

    wire [SRAM_ADDR_W-1:0] load_word_addr = load_byte_addr[SRAM_ADDR_W+1:2];
    wire [1:0] load_byte_sel = load_byte_addr[1:0];
    wire [7:0] load_pixel = cache_data >> (load_byte_sel * 8);
    wire [2:0] load_word_col = load_col[4:2];
    wire [1:0] load_slot = load_row[1:0];
    wire load_last_col = (load_col == w_r - 1'b1);
    wire load_last_ch = (load_ch + 1'b1 >= channels_r);

    reg [31:0] load_pack_next;
    always @* begin
        load_pack_next = load_pack;
        case (load_col[1:0])
            2'd0: load_pack_next[7:0]   = load_pixel;
            2'd1: load_pack_next[15:8]  = load_pixel;
            2'd2: load_pack_next[23:16] = load_pixel;
            default: load_pack_next[31:24] = load_pixel;
        endcase
    end

    wire line_wr_en = (mode_r == MODE_CONV) && (load_state == L_PIXEL) &&
                      (load_last_col || (load_col[1:0] == 2'd3));
    wire [4:0] line_wr_addr = {load_slot, load_word_col};

    wire [1:0] gen_slot = gen_y[1:0] + gen_kh;
    wire [2:0] gen_word_col = gen_group[2:0];
    wire [2:0] row_words = (w_r + 3) >> 2;
    wire [4:0] line_rd_addr = {
        gen_slot,
        ((gen_state == G_RD1) && (gen_word_col + 1'b1 < row_words)) ?
            (gen_word_col + 1'b1) : gen_word_col
    };

    wire [31:0] line0_rd_data;
    wire [31:0] line1_rd_data;
    wire [31:0] line2_rd_data;
    wire [31:0] line3_rd_data;
    wire [31:0] selected_line_data =
        (gen_ch == 3'd0) ? line0_rd_data :
        (gen_ch == 3'd1) ? line1_rd_data :
        (gen_ch == 3'd2) ? line2_rd_data : line3_rd_data;

    wire [63:0] gen_stitch = {gen_word1, gen_word0};
    wire [31:0] gen_raw = gen_stitch >> (gen_kw * 8);
    wire [15:0] group_col = {gen_group[13:0], 2'b00};
    wire [7:0] gen_lane0 = (group_col < out_w_r) ? {~gen_raw[7], gen_raw[6:0]} : 8'b0;
    wire [7:0] gen_lane1 = (group_col + 1 < out_w_r) ? {~gen_raw[15], gen_raw[14:8]} : 8'b0;
    wire [7:0] gen_lane2 = (group_col + 2 < out_w_r) ? {~gen_raw[23], gen_raw[22:16]} : 8'b0;
    wire [7:0] gen_lane3 = (group_col + 3 < out_w_r) ? {~gen_raw[31], gen_raw[30:24]} : 8'b0;
    wire [31:0] conv_fifo_data = {gen_lane3, gen_lane2, gen_lane1, gen_lane0};
    wire [31:0] fc_fifo_data = {24'b0, ~load_pixel[7], load_pixel[6:0]};

    wire conv_push = (mode_r == MODE_CONV) && (gen_state == G_EMIT) && !fifo_full;
    wire fc_push = (mode_r == MODE_FC) && (load_state == L_PIXEL) && !fifo_full;
    wire fifo_push = conv_push || fc_push;
    wire [31:0] fifo_wr_data = fc_push ? fc_fifo_data : conv_fifo_data;

    wire gen_rows_ready = row_valid[gen_y[1:0]] &&
                          row_valid[(gen_y + 1'b1) & 16'h0003] &&
                          row_valid[(gen_y + 2'd2) & 16'h0003] &&
                          (row_tag[gen_y[1:0]] == gen_y) &&
                          (row_tag[(gen_y + 1'b1) & 16'h0003] == gen_y + 1'b1) &&
                          (row_tag[(gen_y + 2'd2) & 16'h0003] == gen_y + 2'd2);

    wire request_can_start = !active && (load_state == L_IDLE) &&
                             (gen_state == G_IDLE) && fifo_empty &&
                             !fifo_out_valid && !fifo_next_valid &&
                             !fifo_rd_pending;

    assign cfg_ready = request_can_start && !req_pending;
    assign sram_rd_en = (load_state == L_REQ);
    assign sram_rd_addr = load_word_addr;
    assign seq_data = fifo_out_data;
    assign seq_valid = fifo_out_valid;

    ActBufferLineDpram32 line0_u (
        .clk(clk),
        .wr_en(line_wr_en && (load_ch == 3'd0)),
        .wr_addr(line_wr_addr), .wr_data(load_pack_next),
        .rd_addr(line_rd_addr), .rd_data(line0_rd_data)
    );
    ActBufferLineDpram32 line1_u (
        .clk(clk),
        .wr_en(line_wr_en && (load_ch == 3'd1)),
        .wr_addr(line_wr_addr), .wr_data(load_pack_next),
        .rd_addr(line_rd_addr), .rd_data(line1_rd_data)
    );
    ActBufferLineDpram32 line2_u (
        .clk(clk),
        .wr_en(line_wr_en && (load_ch == 3'd2)),
        .wr_addr(line_wr_addr), .wr_data(load_pack_next),
        .rd_addr(line_rd_addr), .rd_data(line2_rd_data)
    );
    ActBufferLineDpram32 line3_u (
        .clk(clk),
        .wr_en(line_wr_en && (load_ch == 3'd3)),
        .wr_addr(line_wr_addr), .wr_data(load_pack_next),
        .rd_addr(line_rd_addr), .rd_data(line3_rd_data)
    );

    ActBufferFifoDpram32 #(.ADDR_W(FIFO_ADDR_W)) fifo_u (
        .clk(clk),
        .wr_en(fifo_push), .wr_addr(fifo_wr_ptr), .wr_data(fifo_wr_data),
        .rd_addr(fifo_rd_ptr), .rd_data(fifo_rd_data)
    );

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode_r <= MODE_CONV;
            h_r <= 0;
            w_r <= 0;
            channels_r <= 0;
            channel_stride_r <= 0;
            vector_size_r <= 0;
            out_w_r <= 0;
            out_h_r <= 0;
            groups_r <= 0;
            req_pending <= 1'b0;
            req_base_r <= 0;
            active <= 1'b0;
            load_state <= L_IDLE;
            load_row <= 0;
            load_ch <= 0;
            load_col <= 0;
            load_row_base <= 0;
            load_ch_base <= 0;
            load_byte_addr <= 0;
            load_pack <= 0;
            fc_index <= 0;
            cache_valid <= 1'b0;
            cache_tag <= 0;
            cache_data <= 0;
            row_valid <= 0;
            for (i = 0; i < 4; i = i + 1)
                row_tag[i] <= 0;
            gen_state <= G_IDLE;
            gen_y <= 0;
            gen_group <= 0;
            gen_ch <= 0;
            gen_kh <= 0;
            gen_kw <= 0;
            gen_word0 <= 0;
            gen_word1 <= 0;
            fifo_wr_ptr <= 0;
            fifo_rd_ptr <= 0;
            fifo_count <= 0;
            fifo_out_data <= 0;
            fifo_out_valid <= 1'b0;
            fifo_next_data <= 0;
            fifo_next_valid <= 1'b0;
            fifo_rd_pending <= 1'b0;
        end
        else if (clear) begin
            req_pending <= 1'b0;
            active <= 1'b0;
            load_state <= L_IDLE;
            cache_valid <= 1'b0;
            row_valid <= 0;
            gen_state <= G_IDLE;
            fifo_wr_ptr <= 0;
            fifo_rd_ptr <= 0;
            fifo_count <= 0;
            fifo_out_data <= 0;
            fifo_out_valid <= 1'b0;
            fifo_next_data <= 0;
            fifo_next_valid <= 1'b0;
            fifo_rd_pending <= 1'b0;
        end
        else begin
            if (cfg_valid && cfg_ready) begin
                mode_r <= cfg_mode;
                h_r <= cfg_h;
                w_r <= cfg_w;
                channels_r <= cfg_channels;
                channel_stride_r <= cfg_channel_stride;
                vector_size_r <= cfg_vector_size;
                out_w_r <= cfg_w - 2;
                out_h_r <= cfg_h - 2;
                groups_r <= (cfg_w + 1) >> 2;
            end

            if (req_valid && !req_pending) begin
                req_pending <= 1'b1;
                req_base_r <= req_base_byte;
            end

            if (request_can_start && req_pending) begin
                req_pending <= 1'b0;
                active <= 1'b1;
                cache_valid <= 1'b0;
                fifo_wr_ptr <= 0;
                fifo_rd_ptr <= 0;
                fifo_count <= 0;
                fifo_out_data <= 0;
                fifo_out_valid <= 1'b0;
                fifo_next_data <= 0;
                fifo_next_valid <= 1'b0;
                fifo_rd_pending <= 1'b0;
                if (mode_r == MODE_FC) begin
                    fc_index <= 0;
                    load_byte_addr <= req_base_r;
                    load_state <= L_CHECK;
                end
                else begin
                    load_row <= 0;
                    load_ch <= 0;
                    load_col <= 0;
                    load_row_base <= req_base_r;
                    load_ch_base <= req_base_r;
                    load_byte_addr <= req_base_r;
                    load_pack <= 0;
                    row_valid <= 0;
                    gen_y <= 0;
                    gen_group <= 0;
                    gen_ch <= 0;
                    gen_kh <= 0;
                    gen_kw <= 0;
                    gen_state <= G_IDLE;
                    load_state <= L_CHECK;
                end
            end

            if (load_state == L_CHECK) begin
                if (cache_valid && (cache_tag == load_word_addr))
                    load_state <= L_PIXEL;
                else
                    load_state <= L_REQ;
            end
            else if (load_state == L_REQ) begin
                load_state <= L_WAIT;
            end
            else if (load_state == L_WAIT) begin
                cache_data <= sram_rd_data;
                cache_tag <= load_word_addr;
                cache_valid <= 1'b1;
                load_state <= L_PIXEL;
            end
            else if (load_state == L_PIXEL) begin
                if (mode_r == MODE_FC) begin
                    if (!fifo_full) begin
                        if (fc_index + 1'b1 >= vector_size_r) begin
                            active <= 1'b0;
                            load_state <= L_IDLE;
                        end
                        else begin
                            fc_index <= fc_index + 1'b1;
                            load_byte_addr <= load_byte_addr + 1'b1;
                            load_state <= L_CHECK;
                        end
                    end
                end
                else begin
                    if (load_last_col) begin
                        load_col <= 0;
                        load_pack <= 0;
                        if (load_last_ch) begin
                            row_valid[load_slot] <= 1'b1;
                            row_tag[load_slot] <= load_row;
                            load_ch <= 0;
                            if (load_row + 1'b1 >= h_r) begin
                                load_state <= L_IDLE;
                            end
                            else begin
                                load_row <= load_row + 1'b1;
                                load_row_base <= load_row_base + w_r;
                                load_ch_base <= load_row_base + w_r;
                                load_byte_addr <= load_row_base + w_r;
                                if (load_row + 1'b1 > gen_y + 3)
                                    load_state <= L_IDLE;
                                else
                                    load_state <= L_CHECK;
                            end
                        end
                        else begin
                            load_ch <= load_ch + 1'b1;
                            load_ch_base <= load_ch_base + channel_stride_r;
                            load_byte_addr <= load_ch_base + channel_stride_r;
                            load_state <= L_CHECK;
                        end
                    end
                    else begin
                        load_col <= load_col + 1'b1;
                        load_byte_addr <= load_byte_addr + 1'b1;
                        load_pack <= (load_col[1:0] == 2'd3) ? 32'b0 : load_pack_next;
                        load_state <= L_CHECK;
                    end
                end
            end

            // Once the current output row retires, its slot is free for y+4.
            if ((mode_r == MODE_CONV) && active && (load_state == L_IDLE) &&
                (load_row < h_r) && (load_row <= gen_y + 3)) begin
                row_valid[load_row[1:0]] <= 1'b0;
                load_ch <= 0;
                load_col <= 0;
                load_ch_base <= load_row_base;
                load_byte_addr <= load_row_base;
                load_pack <= 0;
                load_state <= L_CHECK;
            end

            if ((mode_r == MODE_CONV) && active && (gen_state == G_IDLE) &&
                (gen_y < out_h_r) && gen_rows_ready) begin
                gen_group <= 0;
                gen_ch <= 0;
                gen_kh <= 0;
                gen_kw <= 0;
                gen_state <= G_RD0;
            end
            else if (gen_state == G_RD0) begin
                gen_state <= G_RD1;
            end
            else if (gen_state == G_RD1) begin
                gen_word0 <= selected_line_data;
                gen_state <= G_CAP1;
            end
            else if (gen_state == G_CAP1) begin
                gen_word1 <= (gen_word_col + 1'b1 < row_words) ? selected_line_data : 32'b0;
                gen_state <= G_EMIT;
            end

            if (conv_push) begin
                if (gen_kw != 2) begin
                    gen_kw <= gen_kw + 1'b1;
                end
                else begin
                    gen_kw <= 0;
                    if (gen_kh != 2) begin
                        gen_kh <= gen_kh + 1'b1;
                        gen_state <= G_RD0;
                    end
                    else begin
                        gen_kh <= 0;
                        if (gen_ch + 1'b1 < channels_r) begin
                            gen_ch <= gen_ch + 1'b1;
                            gen_state <= G_RD0;
                        end
                        else begin
                            gen_ch <= 0;
                            if (gen_group + 1'b1 < groups_r) begin
                                gen_group <= gen_group + 1'b1;
                                gen_state <= G_RD0;
                            end
                            else begin
                                gen_group <= 0;
                                if (gen_y + 1'b1 >= out_h_r) begin
                                    active <= 1'b0;
                                    gen_state <= G_IDLE;
                                end
                                else begin
                                    row_valid[gen_y[1:0]] <= 1'b0;
                                    gen_y <= gen_y + 1'b1;
                                    gen_state <= G_IDLE;
                                end
                            end
                        end
                    end
                end
            end

            if (fifo_push)
                fifo_wr_ptr <= fifo_wr_ptr + 1'b1;

            if (fifo_pop) begin
                if (fifo_next_valid) begin
                    fifo_out_data <= fifo_next_data;
                    fifo_out_valid <= 1'b1;
                    fifo_next_valid <= 1'b0;
                end
                else begin
                    fifo_out_valid <= 1'b0;
                end
            end

            if (fifo_rd_pending) begin
                if (fifo_out_valid && !fifo_pop) begin
                    fifo_next_data <= fifo_rd_data;
                    fifo_next_valid <= 1'b1;
                end
                else begin
                    fifo_out_data <= fifo_rd_data;
                    fifo_out_valid <= 1'b1;
                end
                fifo_rd_pending <= 1'b0;
            end

            if (fifo_read_issue) begin
                fifo_rd_ptr <= fifo_rd_ptr + 1'b1;
                fifo_rd_pending <= 1'b1;
            end

            case ({fifo_push, fifo_pop})
                2'b10: fifo_count <= fifo_count + 1'b1;
                2'b01: fifo_count <= fifo_count - 1'b1;
                default: fifo_count <= fifo_count;
            endcase
        end
    end

endmodule

module ActBufferLineDpram32 (
    input  wire        clk,
    input  wire        wr_en,
    input  wire [4:0]  wr_addr,
    input  wire [31:0] wr_data,
    input  wire [4:0]  rd_addr,
    output reg  [31:0] rd_data
);
    reg [31:0] mem [0:31];
    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
        rd_data <= mem[rd_addr];
    end
endmodule

module ActBufferFifoDpram32 #(
    parameter ADDR_W = 7
) (
    input  wire              clk,
    input  wire              wr_en,
    input  wire [ADDR_W-1:0] wr_addr,
    input  wire [31:0]       wr_data,
    input  wire [ADDR_W-1:0] rd_addr,
    output reg  [31:0]       rd_data
);
    reg [31:0] mem [0:(1 << ADDR_W)-1];
    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
        rd_data <= mem[rd_addr];
    end
endmodule
