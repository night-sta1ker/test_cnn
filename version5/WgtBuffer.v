`timescale 1ns / 1ps

// Resident packed-weight storage for inference_pure.
// Each word contains four output-channel weights for one K position.
module WgtBuffer #(
    parameter DATA_W = 8,
    parameter ADDR_W = 10,
    parameter DEPTH  = (1 << ADDR_W)
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clear,

    // Model-load port. Writes occur before inference starts.
    input  wire                    wr_en,
    input  wire [ADDR_W-1:0]       wr_addr,
    input  wire [4*DATA_W-1:0]     wr_data,

    // One-cycle request from Sequencer.
    input  wire                    base_valid,
    input  wire [ADDR_W-1:0]       base_addr,
    input  wire [15:0]             k_size,

    output reg  [4*DATA_W-1:0]     stream_data,
    output reg                     stream_valid,
    input  wire                    stream_ready
);

    localparam S_IDLE   = 1'b0;
    localparam S_STREAM = 1'b1;

    reg [4*DATA_W-1:0] mem [0:DEPTH-1];
    reg                 state;
    reg [ADDR_W-1:0]    rd_addr;
    reg [15:0]          words_left;

    // Sequencer has no request-ready signal, so retain one request if needed.
    reg                 req_pending;
    reg [ADDR_W-1:0]    req_base;
    reg [15:0]          req_k_size;

    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            rd_addr      <= {ADDR_W{1'b0}};
            words_left   <= 16'b0;
            stream_data  <= {4*DATA_W{1'b0}};
            stream_valid <= 1'b0;
            req_pending  <= 1'b0;
            req_base     <= {ADDR_W{1'b0}};
            req_k_size   <= 16'b0;
        end
        else if (clear) begin
            state        <= S_IDLE;
            rd_addr      <= {ADDR_W{1'b0}};
            words_left   <= 16'b0;
            stream_data  <= {4*DATA_W{1'b0}};
            stream_valid <= 1'b0;
            req_pending  <= 1'b0;
            req_base     <= {ADDR_W{1'b0}};
            req_k_size   <= 16'b0;
        end
        else begin
            if (base_valid && !req_pending) begin
                req_pending <= 1'b1;
                req_base <= base_addr;
                req_k_size <= k_size;
            end

            if (state == S_IDLE) begin
                stream_valid <= 1'b0;
                if (req_pending) begin
                    req_pending <= 1'b0;
                    if (req_k_size != 16'b0) begin
                        state <= S_STREAM;
                        rd_addr <= req_base;
                        words_left <= req_k_size;
                        stream_data <= mem[req_base];
                        stream_valid <= 1'b1;
                    end
                end
            end
            else if (stream_valid && stream_ready) begin
                if (words_left == 16'd1) begin
                    state <= S_IDLE;
                    words_left <= 16'b0;
                    stream_data <= {4*DATA_W{1'b0}};
                    stream_valid <= 1'b0;
                end
                else begin
                    rd_addr <= rd_addr + 1'b1;
                    words_left <= words_left - 1'b1;
                    stream_data <= mem[rd_addr + 1'b1];
                end
            end
        end
    end

endmodule
