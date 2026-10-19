module hdmi_test_top (
    input  wire sys_clk,      // board clock into MMCM/Clock Wizard
    input  wire rst_btn,      // active high reset

    output wire tmds_clk_p,
    output wire tmds_clk_n,
    output wire [2:0] tmds_data_p,
    output wire [2:0] tmds_data_n
);

    wire pix_clk;
    wire tmds_clk;
    wire clk_locked;

    wire [10:0] hcount;
    wire [9:0]  vcount;
    wire hsync, vsync, de;
    wire [7:0] red, green, blue;

    // Clock Wizard:
    // input: sys_clk
    // outputs:
    //   pix_clk  = 25.175 MHz
    //   tmds_clk = 125.875 MHz
    //   locked
    clk_wiz_hdmi u_clk (
        .clk_in1(sys_clk),
        .reset(rst_btn),
        .clk_out1(pix_clk),
        .clk_out2(tmds_clk),
        .locked(clk_locked)
    );

    video_timing_640x480 u_timing (
        .pix_clk(pix_clk),
        .rst(~clk_locked),
        .hcount(hcount),
        .vcount(vcount),
        .hsync(hsync),
        .vsync(vsync),
        .de(de),
        .frame_start()
    );

    colorbars u_bars (
        .pix_clk(pix_clk),
        .rst(~clk_locked),
        .hcount(hcount),
        .vcount(vcount),
        .de(de),
        .red(red),
        .green(green),
        .blue(blue)
    );

    // Digilent rgb2dvi
    rgb2dvi #(
        .kGenerateSerialClk(1),
        .kClkRange(1),          // set per IP guidance / frequency range
        .kRstActiveHigh(1)
    ) u_rgb2dvi (
        .TMDS_Clk_p(tmds_clk_p),
        .TMDS_Clk_n(tmds_clk_n),
        .TMDS_Data_p(tmds_data_p),
        .TMDS_Data_n(tmds_data_n),

        .aRst(~clk_locked),
        .PixelClk(pix_clk),
        .SerialClk(tmds_clk),

        .vid_pData({red, green, blue}),
        .vid_pVDE(de),
        .vid_pHSync(hsync),
        .vid_pVSync(vsync),

        .ade(), .oSerialClk()
    );

endmodule