module video_timing_640x480 (
    input  wire        pix_clk,
    input  wire        rst,          // active high
    output reg [10:0]  hcount = 0,
    output reg [9:0]   vcount = 0,
    output wire        hsync,
    output wire        vsync,
    output wire        de,
    output wire        frame_start
);

    // 640x480@60 nominal timing
    localparam H_ACTIVE = 640;
    localparam H_FP     = 16;
    localparam H_SYNC   = 96;
    localparam H_BP     = 48;
    localparam H_TOTAL  = H_ACTIVE + H_FP + H_SYNC + H_BP;   // 800

    localparam V_ACTIVE = 480;
    localparam V_FP     = 10;
    localparam V_SYNC   = 2;
    localparam V_BP     = 33;
    localparam V_TOTAL  = V_ACTIVE + V_FP + V_SYNC + V_BP;   // 525

    always @(posedge pix_clk) begin
        if (rst) begin
            hcount <= 0;
            vcount <= 0;
        end else begin
            if (hcount == H_TOTAL - 1) begin
                hcount <= 0;
                if (vcount == V_TOTAL - 1)
                    vcount <= 0;
                else
                    vcount <= vcount + 1;
            end else begin
                hcount <= hcount + 1;
            end
        end
    end

    assign de = (hcount < H_ACTIVE) && (vcount < V_ACTIVE);

    // VGA/480p style negative polarity syncs
    assign hsync = ~((hcount >= H_ACTIVE + H_FP) &&
                     (hcount <  H_ACTIVE + H_FP + H_SYNC));

    assign vsync = ~((vcount >= V_ACTIVE + V_FP) &&
                     (vcount <  V_ACTIVE + V_FP + V_SYNC));

    assign frame_start = (hcount == 0 && vcount == 0);

endmodule