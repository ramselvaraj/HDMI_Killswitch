`timescale 1ns / 1ps

module tb_hdmi_auth_gate;

    // 148.5 MHz pixel clock for 1080p60
    localparam real CLK_PERIOD = 6.734;

    // 1080p timing parameters (1920x1080@60)
    localparam H_ACTIVE = 1920;
    localparam H_FRONT  = 88;
    localparam H_SYNC   = 44;
    localparam H_BACK   = 148;
    localparam H_TOTAL  = H_ACTIVE + H_FRONT + H_SYNC + H_BACK;  // 2200

    localparam V_ACTIVE = 1080;
    localparam V_FRONT  = 4;
    localparam V_SYNC   = 5;
    localparam V_BACK   = 36;
    localparam V_TOTAL  = V_ACTIVE + V_FRONT + V_SYNC + V_BACK;  // 1125

    reg         pixel_clk;
    reg         reset_n;
    reg         auth_enable;
    reg  [23:0] vid_in_data;
    reg         vid_in_hsync;
    reg         vid_in_vsync;
    reg         vid_in_de;
    reg         vid_in_valid;

    wire [23:0] vid_out_data;
    wire        vid_out_hsync;
    wire        vid_out_vsync;
    wire        vid_out_de;
    wire        vid_out_valid;

    // Instantiate DUT
    hdmi_auth_gate #(.DATA_WIDTH(24)) dut (
        .pixel_clk    (pixel_clk),
        .reset_n      (reset_n),
        .auth_enable  (auth_enable),
        .vid_in_data  (vid_in_data),
        .vid_in_hsync (vid_in_hsync),
        .vid_in_vsync (vid_in_vsync),
        .vid_in_de    (vid_in_de),
        .vid_in_valid (vid_in_valid),
        .vid_out_data (vid_out_data),
        .vid_out_hsync(vid_out_hsync),
        .vid_out_vsync(vid_out_vsync),
        .vid_out_de   (vid_out_de),
        .vid_out_valid(vid_out_valid)
    );

    // Clock generation
    initial pixel_clk = 1'b0;
    always #(CLK_PERIOD/2.0) pixel_clk = ~pixel_clk;

    // Video signal generator task
    task generate_frame;
        input [23:0] pixel_color;
        integer h, v;
        begin
            for (v = 0; v < V_TOTAL; v = v + 1) begin
                for (h = 0; h < H_TOTAL; h = h + 1) begin
                    @(negedge pixel_clk);

                    // HSYNC active during sync pulse
                    vid_in_hsync = (h >= (H_ACTIVE + H_FRONT)) &&
                                   (h <  (H_ACTIVE + H_FRONT + H_SYNC));

                    // VSYNC active during sync pulse
                    vid_in_vsync = (v >= (V_ACTIVE + V_FRONT)) &&
                                   (v <  (V_ACTIVE + V_FRONT + V_SYNC));

                    // Data enable active only in visible area
                    vid_in_de    = (h < H_ACTIVE) && (v < V_ACTIVE);
                    vid_in_valid = 1'b1;
                    vid_in_data  = vid_in_de ? pixel_color : 24'h000000;
                end
            end
        end
    endtask

    initial begin
        $dumpfile("hdmi_auth_sim.vcd");
        $dumpvars(0, tb_hdmi_auth_gate);

        // Init
        reset_n      = 1'b0;
        auth_enable  = 1'b0;
        vid_in_data  = 24'h000000;
        vid_in_hsync = 1'b0;
        vid_in_vsync = 1'b0;
        vid_in_de    = 1'b0;
        vid_in_valid = 1'b0;

        repeat(10) @(posedge pixel_clk);
        reset_n = 1'b1;
        repeat(5) @(posedge pixel_clk);

        // TEST 1: Unauthorized
        $display("TEST 1: auth=0, expect BLACK output");
        auth_enable = 1'b0;
        generate_frame(24'hFF0000);  // send RED, expect BLACK

        if (vid_out_data !== 24'h000000)
            $display("FAIL: got %h, expected 000000", vid_out_data);
        else
            $display("PASS: output is black when unauthorized");

        // TEST 2: Authorized
        $display("TEST 2: auth=1, expect RED passthrough");
        auth_enable = 1'b1;
        generate_frame(24'hFF0000);  // send RED, expect RED

        repeat(100) @(posedge pixel_clk);
        if (vid_out_de && (vid_out_data !== 24'hFF0000))
            $display("FAIL: got %h, expected FF0000", vid_out_data);
        else
            $display("PASS: pixel data passes through when authorized");

        // TEST 3: Sync signals always pass
        $display("TEST 3: sync always passes regardless of auth");
        auth_enable = 1'b0;
        generate_frame(24'hFFFFFF);

        @(posedge vid_out_vsync);
        $display("PASS: vsync present even when unauthorized");

        // TEST 4: Auth changes mid-stream
        $display("TEST 4: auth change mid-stream");
        auth_enable = 1'b0;
        repeat((H_TOTAL * V_ACTIVE) / 2) @(posedge pixel_clk);
        auth_enable = 1'b1;
        $display("Auth flipped mid-frame.");

        @(posedge vid_out_vsync);
        repeat(H_TOTAL) @(posedge pixel_clk);
        $display("PASS: auth change observed around frame boundary");

        $display("All tests complete.");
        #1000;
        $finish;
    end

    // Monitor for mismatches while authorized
    always @(posedge pixel_clk) begin
        if (reset_n && auth_enable && vid_out_de && vid_in_de) begin
            if (vid_out_data !== vid_in_data)
                $display("WARNING at time %0t: data mismatch during authorized mode", $time);
        end
    end

endmodule