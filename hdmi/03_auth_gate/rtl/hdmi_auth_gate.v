`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Ram Selvaraj
// 
// Create Date: 04/14/2026 05:45:11 PM
// Design Name: 
// Module Name: hdmi_auth_gate
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

module hdmi_auth_gate #(
    parameter DATA_WIDTH = 24  // RGB888
)(
    input  wire                   pixel_clk,
    input  wire                   reset_n,

    // Authorization control (from AXI GPIO or switch)
    input  wire                   auth_enable,

    // Video input (from DVI2RGB)
    input  wire [DATA_WIDTH-1:0]  vid_in_data,    // RGB pixel data
    input  wire                   vid_in_hsync,
    input  wire                   vid_in_vsync,
    input  wire                   vid_in_de,      // data enable
    input  wire                   vid_in_valid,

    // Video output (to RGB2DVI)
    output reg  [DATA_WIDTH-1:0]  vid_out_data,
    output reg                    vid_out_hsync,
    output reg                    vid_out_vsync,
    output reg                    vid_out_de,
    output reg                    vid_out_valid
);

    // Registered auth signal (avoid glitches mid-frame)
    reg auth_reg;
    reg auth_sync1, auth_sync2;  // 2-stage CDC sync

    // Cross clock domain safely
    always @(posedge pixel_clk or negedge reset_n) begin
        if (!reset_n) begin
            auth_sync1 <= 1'b0;
            auth_sync2 <= 1'b0;
            auth_reg   <= 1'b0;
        end else begin
            auth_sync1 <= auth_enable;
            auth_sync2 <= auth_sync1;
            // Latch only at vsync boundary to prevent torn frames
            if (vid_in_vsync)
                auth_reg <= auth_sync2;
        end
    end

    // Gate logic
    always @(posedge pixel_clk or negedge reset_n) begin
        if (!reset_n) begin
            vid_out_data  <= {DATA_WIDTH{1'b0}};
            vid_out_hsync <= 1'b0;
            vid_out_vsync <= 1'b0;
            vid_out_de    <= 1'b0;
            vid_out_valid <= 1'b0;
        end else begin
            // Always forward sync signals (monitor needs them to stay locked)
            vid_out_hsync <= vid_in_hsync;
            vid_out_vsync <= vid_in_vsync;
            vid_out_de    <= vid_in_de;
            vid_out_valid <= vid_in_valid;

            if (auth_reg) begin
                // AUTHORIZED: pass pixel data through
                vid_out_data <= vid_in_data;
            end else begin
                // UNAUTHORIZED: black screen (but sync intact)
                vid_out_data <= {DATA_WIDTH{1'b0}};
            end
        end
    end

endmodule