`timescale 1ns / 1ps

module postprocess_writer_tb;
    reg clk;
    reg rst_n;
    reg clear;
    reg cfg_valid;
    reg cfg_mode_fc;
    reg [11:0] cfg_result_base_byte;
    reg [5:0] cfg_out_w;
    reg [5:0] cfg_row_stride;
    reg [11:0] cfg_plane_size;
    reg [15:0] cfg_m_size;
    reg [15:0] cfg_n_size;
    reg [127:0] psum_data;
    reg [3:0] psum_valid;
    wire psum_ready;
    reg [15:0] result_m0;
    reg [15:0] result_m1;
    reg [15:0] result_m2;
    reg [15:0] result_m3;
    reg [15:0] result_n_base;
    reg seq_done;
    wire busy;
    wire done;
    wire overflow;
    wire protocol_error;
    wire sram_wr_en;
    wire [9:0] sram_wr_addr;
    wire [31:0] sram_wr_data;
    wire [3:0] sram_wr_be;

    reg param_wr_en;
    reg [4:0] param_wr_addr;
    reg [64:0] param_wr_data;
    reg [4:0] layer_param_base;
    wire [4:0] param_rd_base = layer_param_base + result_n_base[4:0];
    wire [127:0] param_bias;
    wire [63:0] param_mult;
    wire [31:0] param_shift;
    wire [31:0] param_zout;
    wire [3:0] param_relu;
    reg [64:0] param_image [0:21];

    wire [31:0] unused_rd_data;
    integer errors;
    integer i;
    integer ch;
    integer y;
    integer x;
    integer m;
    integer timeout;
    integer stall_cycles;
    reg [7:0] expected;

    ParamBuffer param_u (
        .clk(clk), .wr_en(param_wr_en), .wr_addr(param_wr_addr),
        .wr_data(param_wr_data), .rd_base(param_rd_base),
        .rd_bias(param_bias), .rd_mult(param_mult), .rd_shift(param_shift),
        .rd_zout(param_zout), .rd_relu(param_relu)
    );

    PostProcessWriter #(.QUEUE_ADDR_W(2)) dut (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .cfg_valid(cfg_valid), .cfg_mode_fc(cfg_mode_fc),
        .cfg_result_base_byte(cfg_result_base_byte), .cfg_out_w(cfg_out_w),
        .cfg_row_stride(cfg_row_stride), .cfg_plane_size(cfg_plane_size),
        .cfg_m_size(cfg_m_size), .cfg_n_size(cfg_n_size),
        .psum_data(psum_data), .psum_valid(psum_valid),
        .psum_ready(psum_ready),
        .result_m0(result_m0), .result_m1(result_m1),
        .result_m2(result_m2), .result_m3(result_m3),
        .result_n_base(result_n_base), .seq_done(seq_done),
        .param_bias(param_bias), .param_mult(param_mult),
        .param_shift(param_shift), .param_zout(param_zout),
        .param_relu(param_relu), .busy(busy), .done(done),
        .overflow(overflow), .protocol_error(protocol_error),
        .sram_wr_en(sram_wr_en), .sram_wr_addr(sram_wr_addr),
        .sram_wr_data(sram_wr_data), .sram_wr_be(sram_wr_be)
    );

    ActivationSram sram_u (
        .clk(clk), .rd_en(1'b0), .rd_addr(10'b0), .rd_data(unused_rd_data),
        .wr_en(sram_wr_en), .wr_addr(sram_wr_addr),
        .wr_data(sram_wr_data), .wr_be(sram_wr_be)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    function signed [31:0] psum_value;
        input integer kind;
        input integer channel;
        input integer index;
        integer value;
        begin
            value = ((kind * 313 + channel * 101 + index * 17) % 50000) - 20000;
            psum_value = value;
        end
    endfunction

    function [7:0] golden_quant;
        input integer param_index;
        input signed [31:0] psum;
        reg [64:0] p;
        reg signed [31:0] acc;
        reg signed [63:0] product;
        reg signed [63:0] value;
        begin
            p = param_image[param_index];
            acc = psum + $signed(p[31:0]);
            product = acc * $signed(p[47:32]);
            value = (product >>> (15 + p[55:48])) + $signed({1'b0, p[63:56]});
            if (p[64] && value < $signed({1'b0, p[63:56]}))
                value = $signed({1'b0, p[63:56]});
            if (value < 0)
                golden_quant = 0;
            else if (value > 255)
                golden_quant = 255;
            else
                golden_quant = value[7:0];
        end
    endfunction

    function [7:0] stored_byte;
        input integer byte_addr;
        begin
            stored_byte = sram_u.mem[byte_addr >> 2] >> ((byte_addr & 3) * 8);
        end
    endfunction

    task load_params;
        begin
            for (i = 0; i < 22; i = i + 1) begin
                @(negedge clk);
                param_wr_en = 1'b1;
                param_wr_addr = i[4:0];
                param_wr_data = param_image[i];
            end
            @(negedge clk);
            param_wr_en = 1'b0;
        end
    endtask

    task configure;
        input mode_fc;
        input integer out_w;
        input integer row_stride;
        input integer plane_size;
        input integer m_size;
        input integer n_size;
        input integer param_base;
        begin
            wait (!busy);
            @(negedge clk);
            cfg_mode_fc = mode_fc;
            cfg_result_base_byte = 12'd1024;
            cfg_out_w = out_w[5:0];
            cfg_row_stride = row_stride[5:0];
            cfg_plane_size = plane_size[11:0];
            cfg_m_size = m_size[15:0];
            cfg_n_size = n_size[15:0];
            layer_param_base = param_base[4:0];
            result_n_base = 0;
            cfg_valid = 1'b1;
            @(negedge clk);
            cfg_valid = 1'b0;
        end
    endtask

    task drive_tile;
        input integer n_base;
        input integer m_size;
        input integer n_size;
        input integer kind;
        integer lane;
        reg [3:0] valid_mask;
        begin
            @(negedge clk);
            psum_valid = 0;
            result_n_base = n_base[15:0];
            // Leave one complete cycle for tile-address and ParamBuffer update.
            @(negedge clk);
            for (m = 0; m < m_size; m = m + 1) begin
                valid_mask = 0;
                for (lane = 0; lane < 4; lane = lane + 1) begin
                    if (n_base + lane < n_size) begin
                        valid_mask[lane] = 1'b1;
                        psum_data[lane*32 +: 32] = psum_value(kind, n_base + lane, m);
                    end
                    else begin
                        psum_data[lane*32 +: 32] = 0;
                    end
                end
                result_m0 = m[15:0];
                result_m1 = m[15:0];
                result_m2 = m[15:0];
                result_m3 = m[15:0];
                psum_valid = valid_mask;
                @(posedge clk);
                while (!psum_ready)
                    @(posedge clk);
                @(negedge clk);
            end
            psum_valid = 0;
        end
    endtask

    always @(posedge clk) begin
        if (rst_n && psum_valid && !psum_ready)
            stall_cycles = stall_cycles + 1;
    end

    task finish_layer;
        begin
            @(negedge clk);
            seq_done = 1'b1;
            @(negedge clk);
            seq_done = 1'b0;
            timeout = 0;
            while (!done && timeout < 10000) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            if (!done) begin
                $display("POST FAIL writer done timeout");
                errors = errors + 1;
            end
            if (overflow || protocol_error) begin
                $display("POST FAIL overflow=%b protocol_error=%b", overflow, protocol_error);
                errors = errors + 1;
            end
        end
    endtask

    task check_conv;
        input integer kind;
        input integer channels;
        input integer out_w;
        input integer row_stride;
        input integer out_h;
        input integer plane_size;
        input integer param_base;
        integer byte_addr;
        begin
            for (ch = 0; ch < channels; ch = ch + 1) begin
                for (y = 0; y < out_h; y = y + 1) begin
                    for (x = 0; x < out_w; x = x + 1) begin
                        m = y * row_stride + x;
                        expected = golden_quant(param_base + ch,
                                                psum_value(kind, ch, m));
                        byte_addr = 1024 + ch * plane_size + y * out_w + x;
                        if (stored_byte(byte_addr) !== expected) begin
                            $display("POST FAIL kind=%0d ch=%0d y=%0d x=%0d data=%0d expected=%0d",
                                     kind, ch, y, x, stored_byte(byte_addr), expected);
                            errors = errors + 1;
                        end
                    end
                end
            end
        end
    endtask

    initial begin
        $readmemh("model_params_hw.hex", param_image);
        rst_n = 1'b0;
        clear = 1'b0;
        cfg_valid = 1'b0;
        cfg_mode_fc = 1'b0;
        cfg_result_base_byte = 0;
        cfg_out_w = 0;
        cfg_row_stride = 0;
        cfg_plane_size = 0;
        cfg_m_size = 0;
        cfg_n_size = 0;
        psum_data = 0;
        psum_valid = 0;
        result_m0 = 0;
        result_m1 = 0;
        result_m2 = 0;
        result_m3 = 0;
        result_n_base = 0;
        seq_done = 1'b0;
        param_wr_en = 1'b0;
        param_wr_addr = 0;
        param_wr_data = 0;
        layer_param_base = 0;
        errors = 0;
        stall_cycles = 0;
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        load_params;

        configure(0, 26, 28, 676, 728, 4, 0);
        fork
            begin
                wait (dut.queue_nonempty);
                force dut.queue_pop = 1'b0;
                repeat (24) @(negedge clk);
                release dut.queue_pop;
            end
            begin
                drive_tile(0, 728, 4, 1);
            end
        join
        if (stall_cycles == 0) begin
            $display("POST FAIL forced write stall did not backpressure psum input");
            errors = errors + 1;
        end
        finish_layer;
        check_conv(1, 4, 26, 28, 26, 676, 0);

        configure(0, 11, 12, 121, 132, 8, 4);
        drive_tile(0, 132, 8, 2);
        drive_tile(4, 132, 8, 2);
        finish_layer;
        check_conv(2, 8, 11, 12, 11, 121, 4);

        configure(1, 1, 1, 1, 1, 10, 12);
        drive_tile(0, 1, 10, 3);
        drive_tile(4, 1, 10, 3);
        drive_tile(8, 1, 10, 3);
        finish_layer;
        for (ch = 0; ch < 10; ch = ch + 1) begin
            expected = golden_quant(12 + ch, psum_value(3, ch, 0));
            if (stored_byte(1024 + ch) !== expected) begin
                $display("POST FC FAIL ch=%0d data=%0d expected=%0d",
                         ch, stored_byte(1024 + ch), expected);
                errors = errors + 1;
            end
        end

        if (errors == 0) begin
            $display("POSTPROCESS WRITER TEST PASS");
            $finish;
        end
        else begin
            $display("POSTPROCESS WRITER TEST FAIL errors=%0d", errors);
            $fatal(1, "POSTPROCESS WRITER TEST FAILED");
        end
    end
endmodule
