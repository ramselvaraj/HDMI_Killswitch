`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/18/2026 03:42:52 PM
// Design Name: 
// Module Name: hdmi_auth
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

module hdmi_auth (
    input  wire        sw,             // switch control
    input  wire [23:0] video_in,       // RGB input (8:8:8)
    output wire [23:0] video_out       // RGB output
);

    // White color (R=255, G=255, B=255)
    localparam [23:0] WHITE = 24'b0;

    assign video_out = (sw) ? video_in : WHITE;

endmodule