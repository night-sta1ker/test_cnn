`timescale 1ns / 1ps

// ============================================================================
// SystolicArray4x4 - 4x4 Weight-Stationary systolic array
//
//   Dataflow:
//     activation: left to right, 1 hop/cycle
//     psum:       top to bottom, 1 hop/cycle
//     weight:     stationary after load
//
//   Anti-diagonal weight loading:
//     wgt_ld[phase] loads one cyclic anti-diagonal per cycle.
//
//       phase 0: PE[0][0], PE[1][3], PE[2][2], PE[3][1]
//       phase 1: PE[0][1], PE[1][0], PE[2][3], PE[3][2]
//       phase 2: PE[0][2], PE[1][1], PE[2][0], PE[3][3]
//       phase 3: PE[0][3], PE[1][2], PE[2][1], PE[3][0]
//
//     wgt_in is column-lane ordered: {c3, c2, c1, c0}.
//
//   Mapping:
//     rows = K dimension, cols = N dimension
// ============================================================================

module SystolicArray4x4 #(
    parameter DATA_W = 8,
    parameter ACC_W  = 32
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     en,                 // global clock enable

    // Weight loading - one-hot cyclic anti-diagonal phase
    input  wire [3:0]               wgt_ld,             // load enable per anti-diagonal phase
    input  wire [4*DATA_W-1:0]      wgt_in,             // {c3, c2, c1, c0}

    // Control - global pipeline flush
    input  wire                     clear,

    // Activation input - left boundary, per row
    input  wire [4*DATA_W-1:0]      act_in,             // {r3, r2, r1, r0}

    // Psum control - per-PE mux select, 1 selects zero instead of previous psum.
    // Bit index: row*4 + col.
    input  wire [15:0]              psum_zero,

    // Psum output - selected row boundary, independently per column.
    // psum_out_sel[c*2 +: 2]: 0 = after row0, ..., 3 = after row3.
    input  wire [7:0]               psum_out_sel,
    output wire [4*ACC_W-1:0]       psum_out            // {c3, c2, c1, c0}
);

    // ========================================================================
    // Inter-PE wires
    // ========================================================================
    wire signed [DATA_W-1:0] act_h [0:3][0:4];   // [row][col], col4 = right edge
    wire signed [ACC_W-1:0]  psum_v [0:4][0:3];  // [row][col], row4 = bottom edge
    wire signed [ACC_W-1:0]  psum_to_pe [0:3][0:3];

    // Decoded per-PE control
    wire pe_wgt_ld [0:3][0:3];

    // ========================================================================
    // Control decode: cyclic anti-diagonal phase -> per-PE load signal
    // ========================================================================
    genvar r, c;
    generate
        for (r = 0; r < 4; r = r + 1) begin : gen_ctrl_row
            for (c = 0; c < 4; c = c + 1) begin : gen_ctrl_col
                assign pe_wgt_ld[r][c] = wgt_ld[(r + c) % 4];
            end
        end
    endgenerate

    // ========================================================================
    // Boundary connections
    // ========================================================================
    generate
        for (r = 0; r < 4; r = r + 1) begin : gen_act_bound
            assign act_h[r][0] = act_in[r*DATA_W +: DATA_W];
        end
    endgenerate

    // ========================================================================
    // PE Array
    // ========================================================================
    generate
        for (r = 0; r < 4; r = r + 1) begin : gen_row
            for (c = 0; c < 4; c = c + 1) begin : gen_col
                assign psum_to_pe[r][c] = psum_zero[r*4 + c]
                    ? {ACC_W{1'b0}}
                    : ((r == 0) ? psum_v[4][c] : psum_v[r][c]);

                SystolicPE #(
                    .DATA_W(DATA_W),
                    .ACC_W (ACC_W)
                ) pe_u (
                    .clk     (clk),
                    .rst_n   (rst_n),
                    .wgt_ld  (pe_wgt_ld[r][c]),
                    .wgt_in  (wgt_in[c*DATA_W +: DATA_W]),
                    .en      (en),
                    .clear   (clear),
                    .act_in  (act_h[r][c]),
                    .psum_in (psum_to_pe[r][c]),
                    .act_out (act_h[r][c+1]),
                    .psum_out(psum_v[r+1][c])
                );

            end
        end
    endgenerate

    // ========================================================================
    // Output mux
    //   psum_out[c] = selected row boundary after row psum_out_sel, column c.
    // ========================================================================
    generate
        for (c = 0; c < 4; c = c + 1) begin : gen_out_col
            reg signed [ACC_W-1:0] psum_mux;

            always @(*) begin
                case (psum_out_sel[c*2 +: 2])
                    2'd0: psum_mux = psum_v[1][c];
                    2'd1: psum_mux = psum_v[2][c];
                    2'd2: psum_mux = psum_v[3][c];
                    2'd3: psum_mux = psum_v[4][c];
                endcase
            end

            assign psum_out[c*ACC_W +: ACC_W] = psum_mux;
        end
    endgenerate

endmodule
