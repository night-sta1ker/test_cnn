`timescale 1ns / 1ps

// ============================================================================
// SystolicPE — Weight-Stationary systolic processing element
//
//   Dataflow:  activation → (left to right, 1 hop/cycle)
//              psum       → (top to bottom, 1 hop/cycle)
//              weight     → stationary (pre-loaded, held in local register)
//
//   Operation: psum_out = psum_in + act_in * wgt_stored
//              act_out   = act_in (registered)
//
//   Timing:    1 cycle per PE hop (output-registered, classic systolic)
//              combinational path = 1 mul + 1 add → ~20 logic levels
// ============================================================================

module SystolicPE #(
    parameter DATA_W = 8,
    parameter ACC_W  = 32
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Weight loading
    input  wire                     wgt_ld,
    input  wire signed [DATA_W-1:0] wgt_in,

    // Control
    input  wire                     en,          // clock enable (freeze when low)
    input  wire                     clear,       // flush psum pipeline

    // Systolic dataflow ports
    input  wire signed [DATA_W-1:0] act_in,      // from left neighbor
    input  wire signed [ACC_W-1:0]  psum_in,     // from top  neighbor
    output wire signed [DATA_W-1:0] act_out,     // to   right neighbor
    output wire signed [ACC_W-1:0]  psum_out     // to   bottom neighbor
);

    // ---- local weight (stationary) ------------------------------------------
    reg signed [DATA_W-1:0] wgt;

    // ---- systolic pipeline registers ----------------------------------------
    reg signed [DATA_W-1:0] act_d;        // activation delay
    reg signed [ACC_W-1:0]  psum_out_reg; // psum output register

    // ---- combinational datapath ---------------------------------------------
    localparam PROD_W = 2 * DATA_W;

    wire signed [PROD_W-1:0] product   = act_in * wgt;
    wire signed [ACC_W-1:0]  psum_next = clear
        ? {ACC_W{1'b0}}
        : psum_in + {{(ACC_W - PROD_W){product[PROD_W-1]}}, product};

    // ---- output assignments -------------------------------------------------
    assign act_out  = act_d;
    assign psum_out = psum_out_reg;

    // ---- sequential ---------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wgt          <= {DATA_W{1'b0}};
            act_d        <= {DATA_W{1'b0}};
            psum_out_reg <= {ACC_W{1'b0}};
        end else if (en) begin
            if (wgt_ld)
                wgt <= wgt_in;

            act_d        <= clear ? {DATA_W{1'b0}} : act_in;
            psum_out_reg <= psum_next;
        end
    end

endmodule
