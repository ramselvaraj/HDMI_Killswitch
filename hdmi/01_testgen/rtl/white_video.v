`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/16/2026 10:28:08 PM
// Design Name: 
// Module Name: white_video
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module white_video (
    input  wire        pixel_clk,
    input  wire        rst,
    output reg  [23:0] vid_pData,
    output reg         vid_pVDE,
    output reg         vid_pHSync,
    output reg         vid_pVSync
);

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

    reg [10:0] hcount = 0;
    reg [9:0]  vcount = 0;

    wire h_active = (hcount < H_ACTIVE);
    wire v_active = (vcount < V_ACTIVE);
    wire active_video = h_active && v_active;

    always @(posedge pixel_clk) begin
        if (rst) begin
            hcount     <= 0;
            vcount     <= 0;
            vid_pData  <= 24'h000000;
            vid_pVDE   <= 1'b0;
            vid_pHSync <= 1'b1;
            vid_pVSync <= 1'b1;
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

            vid_pVDE <= active_video;

            // VGA/CEA-style active-low syncs for 640x480
            vid_pHSync <= ~((hcount >= (H_ACTIVE + H_FP)) &&
                            (hcount <  (H_ACTIVE + H_FP + H_SYNC)));

            vid_pVSync <= ~((vcount >= (V_ACTIVE + V_FP)) &&
                            (vcount <  (V_ACTIVE + V_FP + V_SYNC)));

            if (active_video)
                vid_pData <= 24'hFFFFFF;   // solid white
            else
                vid_pData <= 24'h000000;   // black during blanking
        end
    end

endmodule