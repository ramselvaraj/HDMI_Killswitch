//module auth_overlay #
//(
//    parameter integer H_ACTIVE = 1280,
//    parameter integer V_ACTIVE = 720,

//    parameter integer BANNER_Y0 = 16,
//    parameter integer PAD_X     = 16,
//    parameter integer PAD_Y     = 8,
//    parameter integer BORDER    = 2,
//    parameter integer CHAR_W    = 8,
//    parameter integer CHAR_H    = 8
//)
//(
//    input  wire        PixelClk,
//    input  wire        rst,          // sync reset in PixelClk domain
//    input  wire        auth_sw,      // external switch (async -> synchronized below)

//    input  wire [23:0] s_vid_pData,
//    input  wire        s_vid_pVDE,
//    input  wire        s_vid_pHSync,
//    input  wire        s_vid_pVSync,

//    output wire [23:0] m_vid_pData,
//    output wire        m_vid_pVDE,
//    output wire        m_vid_pHSync,
//    output wire        m_vid_pVSync
//);

//    // ------------------------------------------------------------------------
//    // Digilent video bus packing is {R, B, G}, not {R, G, B}
//    // ------------------------------------------------------------------------
//    localparam [23:0] C_BLACK     = {8'h00, 8'h00, 8'h00};
//    localparam [23:0] C_WHITE     = {8'hFF, 8'hFF, 8'hFF};

//    localparam [23:0] C_AUTH_BG   = {8'h18, 8'h18, 8'hA0}; // dark green
//    localparam [23:0] C_NOAUTH_BG = {8'hA0, 8'h18, 8'h18}; // dark red

//    localparam integer AUTH_LEN   = 10; // "AUTHORIZED"
//    localparam integer NOAUTH_LEN = 14; // "NOT AUTHORIZED"

//    localparam integer AUTH_TEXT_W   = AUTH_LEN   * CHAR_W;
//    localparam integer NOAUTH_TEXT_W = NOAUTH_LEN * CHAR_W;

//    localparam integer AUTH_BANNER_W   = AUTH_TEXT_W   + (2 * PAD_X);
//    localparam integer NOAUTH_BANNER_W = NOAUTH_TEXT_W + (2 * PAD_X);
//    localparam integer BANNER_H        = CHAR_H + (2 * PAD_Y);

//    // ------------------------------------------------------------------------
//    // Synchronize the switch into PixelClk domain
//    // ------------------------------------------------------------------------
//    reg auth_meta = 1'b0;
//    reg auth_sync = 1'b0;

//    always @(posedge PixelClk) begin
//        if (rst) begin
//            auth_meta <= 1'b0;
//            auth_sync <= 1'b0;
//        end else begin
//            auth_meta <= auth_sw;
//            auth_sync <= auth_meta;
//        end
//    end

//    // ------------------------------------------------------------------------
//    // Active-video x/y counters
//    //
//    // x/y are coordinates inside the active picture only.
//    // They assume a fixed mode (H_ACTIVE x V_ACTIVE).
//    // ------------------------------------------------------------------------
//    reg [11:0] x = 12'd0;
//    reg [11:0] y = 12'd0;

//    reg prev_vde   = 1'b0;
//    reg prev_vsync = 1'b0;
//    reg seen_first_active_line = 1'b0;

//    wire vde_rise   =  s_vid_pVDE & ~prev_vde;
//    wire vsync_edge =  s_vid_pVSync ^ prev_vsync;

//    always @(posedge PixelClk) begin
//        if (rst) begin
//            x <= 12'd0;
//            y <= 12'd0;
//            prev_vde <= 1'b0;
//            prev_vsync <= 1'b0;
//            seen_first_active_line <= 1'b0;
//        end else begin
//            prev_vde   <= s_vid_pVDE;
//            prev_vsync <= s_vid_pVSync;

//            // Reset active-line tracking on any VSync edge.
//            // Since VSync edges happen during blanking, this is fine.
//            if (vsync_edge) begin
//                y <= 12'd0;
//                seen_first_active_line <= 1'b0;
//            end

//            if (vde_rise) begin
//                x <= 12'd0;

//                if (!seen_first_active_line) begin
//                    y <= 12'd0;
//                    seen_first_active_line <= 1'b1;
//                end else if (y < (V_ACTIVE - 1)) begin
//                    y <= y + 12'd1;
//                end
//            end else if (s_vid_pVDE) begin
//                if (x < (H_ACTIVE - 1))
//                    x <= x + 12'd1;
//            end
//        end
//    end

//    // ------------------------------------------------------------------------
//    // Message selection / geometry
//    // ------------------------------------------------------------------------
//    wire        auth_mode   = auth_sync;
//    wire [11:0] text_w      = auth_mode ? AUTH_TEXT_W[11:0]   : NOAUTH_TEXT_W[11:0];
//    wire [11:0] banner_w    = auth_mode ? AUTH_BANNER_W[11:0] : NOAUTH_BANNER_W[11:0];
//    wire [11:0] banner_x0   = (H_ACTIVE > banner_w) ? ((H_ACTIVE - banner_w) >> 1) : 12'd0;
//    wire [23:0] banner_fill = auth_mode ? C_AUTH_BG : C_NOAUTH_BG;

//    // ------------------------------------------------------------------------
//    // Banner / text region tests
//    // ------------------------------------------------------------------------
//    wire in_banner =
//        s_vid_pVDE &&
//        (x >= banner_x0) &&
//        (x <  banner_x0 + banner_w) &&
//        (y >= BANNER_Y0) &&
//        (y <  BANNER_Y0 + BANNER_H);

//    wire [11:0] rel_x = x - banner_x0;
//    wire [11:0] rel_y = y - BANNER_Y0;

//    wire in_border =
//        in_banner &&
//        (
//            (rel_x < BORDER) ||
//            (rel_x >= (banner_w - BORDER)) ||
//            (rel_y < BORDER) ||
//            (rel_y >= (BANNER_H - BORDER))
//        );

//    wire in_text_box =
//        in_banner &&
//        (rel_x >= PAD_X) &&
//        (rel_x <  PAD_X + text_w) &&
//        (rel_y >= PAD_Y) &&
//        (rel_y <  PAD_Y + CHAR_H);

//    wire [11:0] text_x = rel_x - PAD_X;
//    wire [11:0] text_y = rel_y - PAD_Y;

//    wire [4:0] char_index = text_x[11:3]; // divide by 8
//    wire [2:0] char_col   = text_x[2:0];  // modulo 8
//    wire [2:0] char_row   = text_y[2:0];

//    // ------------------------------------------------------------------------
//    // Message ROM
//    // ------------------------------------------------------------------------
//    function [7:0] msg_char;
//        input auth;
//        input [4:0] idx;
//        begin
//            if (auth) begin
//                case (idx)
//                    5'd0: msg_char = "A";
//                    5'd1: msg_char = "U";
//                    5'd2: msg_char = "T";
//                    5'd3: msg_char = "H";
//                    5'd4: msg_char = "O";
//                    5'd5: msg_char = "R";
//                    5'd6: msg_char = "I";
//                    5'd7: msg_char = "Z";
//                    5'd8: msg_char = "E";
//                    5'd9: msg_char = "D";
//                    default: msg_char = " ";
//                endcase
//            end else begin
//                case (idx)
//                    5'd0:  msg_char = "N";
//                    5'd1:  msg_char = "O";
//                    5'd2:  msg_char = "T";
//                    5'd3:  msg_char = " ";
//                    5'd4:  msg_char = "A";
//                    5'd5:  msg_char = "U";
//                    5'd6:  msg_char = "T";
//                    5'd7:  msg_char = "H";
//                    5'd8:  msg_char = "O";
//                    5'd9:  msg_char = "R";
//                    5'd10: msg_char = "I";
//                    5'd11: msg_char = "Z";
//                    5'd12: msg_char = "E";
//                    5'd13: msg_char = "D";
//                    default: msg_char = " ";
//                endcase
//            end
//        end
//    endfunction

//    // ------------------------------------------------------------------------
//    // 8x8 glyph ROM for needed characters only
//    // bit[7] = leftmost pixel
//    // ------------------------------------------------------------------------
//    function [7:0] glyph_row;
//        input [7:0] ch;
//        input [2:0] row;
//        begin
//            case (ch)
//                "A": begin
//                    case (row)
//                        3'd0: glyph_row = 8'b00011000;
//                        3'd1: glyph_row = 8'b00100100;
//                        3'd2: glyph_row = 8'b01000010;
//                        3'd3: glyph_row = 8'b01000010;
//                        3'd4: glyph_row = 8'b01111110;
//                        3'd5: glyph_row = 8'b01000010;
//                        3'd6: glyph_row = 8'b01000010;
//                        default: glyph_row = 8'b00000000;
//                    endcase
//                end

//                "D": begin
//                    case (row)
//                        3'd0: glyph_row = 8'b01111000;
//                        3'd1: glyph_row = 8'b01000100;
//                        3'd2: glyph_row = 8'b01000010;
//                        3'd3: glyph_row = 8'b01000010;
//                        3'd4: glyph_row = 8'b01000010;
//                        3'd5: glyph_row = 8'b01000100;
//                        3'd6: glyph_row = 8'b01111000;
//                        default: glyph_row = 8'b00000000;
//                    endcase
//                end

//                "E": begin
//                    case (row)
//                        3'd0: glyph_row = 8'b01111110;
//                        3'd1: glyph_row = 8'b01000000;
//                        3'd2: glyph_row = 8'b01000000;
//                        3'd3: glyph_row = 8'b01111100;
//                        3'd4: glyph_row = 8'b01000000;
//                        3'd5: glyph_row = 8'b01000000;
//                        3'd6: glyph_row = 8'b01111110;
//                        default: glyph_row = 8'b00000000;
//                    endcase
//                end

//                "H": begin
//                    case (row)
//                        3'd0: glyph_row = 8'b01000010;
//                        3'd1: glyph_row = 8'b01000010;
//                        3'd2: glyph_row = 8'b01000010;
//                        3'd3: glyph_row = 8'b01111110;
//                        3'd4: glyph_row = 8'b01000010;
//                        3'd5: glyph_row = 8'b01000010;
//                        3'd6: glyph_row = 8'b01000010;
//                        default: glyph_row = 8'b00000000;
//                    endcase
//                end

//                "I": begin
//                    case (row)
//                        3'd0: glyph_row = 8'b00111100;
//                        3'd1: glyph_row = 8'b00011000;
//                        3'd2: glyph_row = 8'b00011000;
//                        3'd3: glyph_row = 8'b00011000;
//                        3'd4: glyph_row = 8'b00011000;
//                        3'd5: glyph_row = 8'b00011000;
//                        3'd6: glyph_row = 8'b00111100;
//                        default: glyph_row = 8'b00000000;
//                    endcase
//                end

//                "N": begin
//                    case (row)
//                        3'd0: glyph_row = 8'b01000010;
//                        3'd1: glyph_row = 8'b01100010;
//                        3'd2: glyph_row = 8'b01010010;
//                        3'd3: glyph_row = 8'b01001010;
//                        3'd4: glyph_row = 8'b01000110;
//                        3'd5: glyph_row = 8'b01000010;
//                        3'd6: glyph_row = 8'b01000010;
//                        default: glyph_row = 8'b00000000;
//                    endcase
//                end

//                "O": begin
//                    case (row)
//                        3'd0: glyph_row = 8'b00111100;
//                        3'd1: glyph_row = 8'b01000010;
//                        3'd2: glyph_row = 8'b01000010;
//                        3'd3: glyph_row = 8'b01000010;
//                        3'd4: glyph_row = 8'b01000010;
//                        3'd5: glyph_row = 8'b01000010;
//                        3'd6: glyph_row = 8'b00111100;
//                        default: glyph_row = 8'b00000000;
//                    endcase
//                end

//                "R": begin
//                    case (row)
//                        3'd0: glyph_row = 8'b01111100;
//                        3'd1: glyph_row = 8'b01000010;
//                        3'd2: glyph_row = 8'b01000010;
//                        3'd3: glyph_row = 8'b01111100;
//                        3'd4: glyph_row = 8'b01001000;
//                        3'd5: glyph_row = 8'b01000100;
//                        3'd6: glyph_row = 8'b01000010;
//                        default: glyph_row = 8'b00000000;
//                    endcase
//                end

//                "T": begin
//                    case (row)
//                        3'd0: glyph_row = 8'b01111110;
//                        3'd1: glyph_row = 8'b00011000;
//                        3'd2: glyph_row = 8'b00011000;
//                        3'd3: glyph_row = 8'b00011000;
//                        3'd4: glyph_row = 8'b00011000;
//                        3'd5: glyph_row = 8'b00011000;
//                        3'd6: glyph_row = 8'b00011000;
//                        default: glyph_row = 8'b00000000;
//                    endcase
//                end

//                "U": begin
//                    case (row)
//                        3'd0: glyph_row = 8'b01000010;
//                        3'd1: glyph_row = 8'b01000010;
//                        3'd2: glyph_row = 8'b01000010;
//                        3'd3: glyph_row = 8'b01000010;
//                        3'd4: glyph_row = 8'b01000010;
//                        3'd5: glyph_row = 8'b01000010;
//                        3'd6: glyph_row = 8'b00111100;
//                        default: glyph_row = 8'b00000000;
//                    endcase
//                end

//                "Z": begin
//                    case (row)
//                        3'd0: glyph_row = 8'b01111110;
//                        3'd1: glyph_row = 8'b00000100;
//                        3'd2: glyph_row = 8'b00001000;
//                        3'd3: glyph_row = 8'b00010000;
//                        3'd4: glyph_row = 8'b00100000;
//                        3'd5: glyph_row = 8'b01000000;
//                        3'd6: glyph_row = 8'b01111110;
//                        default: glyph_row = 8'b00000000;
//                    endcase
//                end

//                " ": glyph_row = 8'b00000000;

//                default: glyph_row = 8'b00000000;
//            endcase
//        end
//    endfunction

//    wire [7:0] cur_char  = msg_char(auth_mode, char_index);
//    wire [7:0] cur_glyph = glyph_row(cur_char, char_row);

//    wire text_pixel =
//        in_text_box &&
//        (char_index < (auth_mode ? AUTH_LEN : NOAUTH_LEN)) &&
//        cur_glyph[7 - char_col];

//    // ------------------------------------------------------------------------
//    // Pass timing straight through; modify only pixel data
//    // ------------------------------------------------------------------------
//    assign m_vid_pVDE   = s_vid_pVDE;
//    assign m_vid_pHSync = s_vid_pHSync;
//    assign m_vid_pVSync = s_vid_pVSync;

//    assign m_vid_pData =
//        (!s_vid_pVDE) ? C_BLACK :
//        (in_banner)   ? (in_border ? C_WHITE :
//                         text_pixel ? C_WHITE :
//                         banner_fill)
//                      : (auth_mode ? s_vid_pData : C_BLACK);

//endmodule

module auth_overlay #
(
    parameter integer H_ACTIVE = 1280,
    parameter integer V_ACTIVE = 720,

    parameter integer BANNER_Y0 = 16,
    parameter integer PAD_X     = 16,
    parameter integer PAD_Y     = 8,
    parameter integer BORDER    = 2,

    // These now control the actual rendered character size
    parameter integer CHAR_W    = 16,
    parameter integer CHAR_H    = 16
)
(
    input  wire        PixelClk,
    input  wire        rst,          // sync reset in PixelClk domain
    input  wire        auth_sw,      // external switch (async -> synchronized below)

    input  wire [23:0] s_vid_pData,
    input  wire        s_vid_pVDE,
    input  wire        s_vid_pHSync,
    input  wire        s_vid_pVSync,

    output wire [23:0] m_vid_pData,
    output wire        m_vid_pVDE,
    output wire        m_vid_pHSync,
    output wire        m_vid_pVSync
);

    localparam integer GLYPH_W = 8;
    localparam integer GLYPH_H = 8;

    // ------------------------------------------------------------------------
    // Digilent video bus packing is {R, B, G}, not {R, G, B}
    // ------------------------------------------------------------------------
    localparam [23:0] C_BLACK     = {8'h00, 8'h00, 8'h00};
    localparam [23:0] C_WHITE     = {8'hFF, 8'hFF, 8'hFF};

    localparam [23:0] C_AUTH_BG   = {8'h18, 8'h18, 8'hA0}; // dark green
    localparam [23:0] C_NOAUTH_BG = {8'hA0, 8'h18, 8'h18}; // dark red

    localparam integer AUTH_LEN   = 10; // "AUTHORIZED"
    localparam integer NOAUTH_LEN = 14; // "NOT AUTHORIZED"

    localparam integer AUTH_TEXT_W   = AUTH_LEN   * CHAR_W;
    localparam integer NOAUTH_TEXT_W = NOAUTH_LEN * CHAR_W;

    localparam integer AUTH_BANNER_W   = AUTH_TEXT_W   + (2 * PAD_X);
    localparam integer NOAUTH_BANNER_W = NOAUTH_TEXT_W + (2 * PAD_X);
    localparam integer BANNER_H        = CHAR_H + (2 * PAD_Y);

    // ------------------------------------------------------------------------
    // Synchronize the switch into PixelClk domain
    // ------------------------------------------------------------------------
    reg auth_meta = 1'b0;
    reg auth_sync = 1'b0;

    always @(posedge PixelClk) begin
        if (rst) begin
            auth_meta <= 1'b0;
            auth_sync <= 1'b0;
        end else begin
            auth_meta <= auth_sw;
            auth_sync <= auth_meta;
        end
    end

    // ------------------------------------------------------------------------
    // Active-video x/y counters
    // ------------------------------------------------------------------------
    reg [11:0] x = 12'd0;
    reg [11:0] y = 12'd0;

    reg prev_vde   = 1'b0;
    reg prev_vsync = 1'b0;
    reg seen_first_active_line = 1'b0;

    wire vde_rise   =  s_vid_pVDE & ~prev_vde;
    wire vsync_edge =  s_vid_pVSync ^ prev_vsync;

    always @(posedge PixelClk) begin
        if (rst) begin
            x <= 12'd0;
            y <= 12'd0;
            prev_vde <= 1'b0;
            prev_vsync <= 1'b0;
            seen_first_active_line <= 1'b0;
        end else begin
            prev_vde   <= s_vid_pVDE;
            prev_vsync <= s_vid_pVSync;

            if (vsync_edge) begin
                y <= 12'd0;
                seen_first_active_line <= 1'b0;
            end

            if (vde_rise) begin
                x <= 12'd0;

                if (!seen_first_active_line) begin
                    y <= 12'd0;
                    seen_first_active_line <= 1'b1;
                end else if (y < (V_ACTIVE - 1)) begin
                    y <= y + 12'd1;
                end
            end else if (s_vid_pVDE) begin
                if (x < (H_ACTIVE - 1))
                    x <= x + 12'd1;
            end
        end
    end

    // ------------------------------------------------------------------------
    // Message selection / geometry
    // ------------------------------------------------------------------------
    wire        auth_mode   = auth_sync;
    wire [11:0] text_w      = auth_mode ? AUTH_TEXT_W[11:0]   : NOAUTH_TEXT_W[11:0];
    wire [11:0] banner_w    = auth_mode ? AUTH_BANNER_W[11:0] : NOAUTH_BANNER_W[11:0];
    wire [11:0] banner_x0   = (H_ACTIVE > banner_w) ? ((H_ACTIVE - banner_w) >> 1) : 12'd0;
    wire [23:0] banner_fill = auth_mode ? C_AUTH_BG : C_NOAUTH_BG;

    // ------------------------------------------------------------------------
    // Banner / text region tests
    // ------------------------------------------------------------------------
    wire in_banner =
        s_vid_pVDE &&
        (x >= banner_x0) &&
        (x <  banner_x0 + banner_w) &&
        (y >= BANNER_Y0) &&
        (y <  BANNER_Y0 + BANNER_H);

    wire [11:0] rel_x = x - banner_x0;
    wire [11:0] rel_y = y - BANNER_Y0;

    wire in_border =
        in_banner &&
        (
            (rel_x < BORDER) ||
            (rel_x >= (banner_w - BORDER)) ||
            (rel_y < BORDER) ||
            (rel_y >= (BANNER_H - BORDER))
        );

    wire in_text_box =
        in_banner &&
        (rel_x >= PAD_X) &&
        (rel_x <  PAD_X + text_w) &&
        (rel_y >= PAD_Y) &&
        (rel_y <  PAD_Y + CHAR_H);

    wire [11:0] text_x = rel_x - PAD_X;
    wire [11:0] text_y = rel_y - PAD_Y;

    // Character index in the string
    wire [4:0] char_index = text_x / CHAR_W;

    // Pixel position within the rendered character cell
    wire [11:0] char_cell_x = text_x % CHAR_W;
    wire [11:0] char_cell_y = text_y; // only one row of text vertically

    // Map scaled character cell back into the original 8x8 glyph space
    wire [15:0] glyph_x_num = char_cell_x << 3; // * 8
    wire [15:0] glyph_y_num = char_cell_y << 3; // * 8

    wire [2:0] char_col = glyph_x_num / CHAR_W; // 0..7
    wire [2:0] char_row = glyph_y_num / CHAR_H; // 0..7

    // ------------------------------------------------------------------------
    // Message ROM
    // ------------------------------------------------------------------------
    function [7:0] msg_char;
        input auth;
        input [4:0] idx;
        begin
            if (auth) begin
                case (idx)
                    5'd0: msg_char = "A";
                    5'd1: msg_char = "U";
                    5'd2: msg_char = "T";
                    5'd3: msg_char = "H";
                    5'd4: msg_char = "O";
                    5'd5: msg_char = "R";
                    5'd6: msg_char = "I";
                    5'd7: msg_char = "Z";
                    5'd8: msg_char = "E";
                    5'd9: msg_char = "D";
                    default: msg_char = " ";
                endcase
            end else begin
                case (idx)
                    5'd0:  msg_char = "N";
                    5'd1:  msg_char = "O";
                    5'd2:  msg_char = "T";
                    5'd3:  msg_char = " ";
                    5'd4:  msg_char = "A";
                    5'd5:  msg_char = "U";
                    5'd6:  msg_char = "T";
                    5'd7:  msg_char = "H";
                    5'd8:  msg_char = "O";
                    5'd9:  msg_char = "R";
                    5'd10: msg_char = "I";
                    5'd11: msg_char = "Z";
                    5'd12: msg_char = "E";
                    5'd13: msg_char = "D";
                    default: msg_char = " ";
                endcase
            end
        end
    endfunction

    // ------------------------------------------------------------------------
    // 8x8 glyph ROM for needed characters only
    // bit[7] = leftmost pixel
    // ------------------------------------------------------------------------
    function [7:0] glyph_row;
        input [7:0] ch;
        input [2:0] row;
        begin
            case (ch)
                "A": begin
                    case (row)
                        3'd0: glyph_row = 8'b00011000;
                        3'd1: glyph_row = 8'b00100100;
                        3'd2: glyph_row = 8'b01000010;
                        3'd3: glyph_row = 8'b01000010;
                        3'd4: glyph_row = 8'b01111110;
                        3'd5: glyph_row = 8'b01000010;
                        3'd6: glyph_row = 8'b01000010;
                        default: glyph_row = 8'b00000000;
                    endcase
                end

                "D": begin
                    case (row)
                        3'd0: glyph_row = 8'b01111000;
                        3'd1: glyph_row = 8'b01000100;
                        3'd2: glyph_row = 8'b01000010;
                        3'd3: glyph_row = 8'b01000010;
                        3'd4: glyph_row = 8'b01000010;
                        3'd5: glyph_row = 8'b01000100;
                        3'd6: glyph_row = 8'b01111000;
                        default: glyph_row = 8'b00000000;
                    endcase
                end

                "E": begin
                    case (row)
                        3'd0: glyph_row = 8'b01111110;
                        3'd1: glyph_row = 8'b01000000;
                        3'd2: glyph_row = 8'b01000000;
                        3'd3: glyph_row = 8'b01111100;
                        3'd4: glyph_row = 8'b01000000;
                        3'd5: glyph_row = 8'b01000000;
                        3'd6: glyph_row = 8'b01111110;
                        default: glyph_row = 8'b00000000;
                    endcase
                end

                "H": begin
                    case (row)
                        3'd0: glyph_row = 8'b01000010;
                        3'd1: glyph_row = 8'b01000010;
                        3'd2: glyph_row = 8'b01000010;
                        3'd3: glyph_row = 8'b01111110;
                        3'd4: glyph_row = 8'b01000010;
                        3'd5: glyph_row = 8'b01000010;
                        3'd6: glyph_row = 8'b01000010;
                        default: glyph_row = 8'b00000000;
                    endcase
                end

                "I": begin
                    case (row)
                        3'd0: glyph_row = 8'b00111100;
                        3'd1: glyph_row = 8'b00011000;
                        3'd2: glyph_row = 8'b00011000;
                        3'd3: glyph_row = 8'b00011000;
                        3'd4: glyph_row = 8'b00011000;
                        3'd5: glyph_row = 8'b00011000;
                        3'd6: glyph_row = 8'b00111100;
                        default: glyph_row = 8'b00000000;
                    endcase
                end

                "N": begin
                    case (row)
                        3'd0: glyph_row = 8'b01000010;
                        3'd1: glyph_row = 8'b01100010;
                        3'd2: glyph_row = 8'b01010010;
                        3'd3: glyph_row = 8'b01001010;
                        3'd4: glyph_row = 8'b01000110;
                        3'd5: glyph_row = 8'b01000010;
                        3'd6: glyph_row = 8'b01000010;
                        default: glyph_row = 8'b00000000;
                    endcase
                end

                "O": begin
                    case (row)
                        3'd0: glyph_row = 8'b00111100;
                        3'd1: glyph_row = 8'b01000010;
                        3'd2: glyph_row = 8'b01000010;
                        3'd3: glyph_row = 8'b01000010;
                        3'd4: glyph_row = 8'b01000010;
                        3'd5: glyph_row = 8'b01000010;
                        3'd6: glyph_row = 8'b00111100;
                        default: glyph_row = 8'b00000000;
                    endcase
                end

                "R": begin
                    case (row)
                        3'd0: glyph_row = 8'b01111100;
                        3'd1: glyph_row = 8'b01000010;
                        3'd2: glyph_row = 8'b01000010;
                        3'd3: glyph_row = 8'b01111100;
                        3'd4: glyph_row = 8'b01001000;
                        3'd5: glyph_row = 8'b01000100;
                        3'd6: glyph_row = 8'b01000010;
                        default: glyph_row = 8'b00000000;
                    endcase
                end

                "T": begin
                    case (row)
                        3'd0: glyph_row = 8'b01111110;
                        3'd1: glyph_row = 8'b00011000;
                        3'd2: glyph_row = 8'b00011000;
                        3'd3: glyph_row = 8'b00011000;
                        3'd4: glyph_row = 8'b00011000;
                        3'd5: glyph_row = 8'b00011000;
                        3'd6: glyph_row = 8'b00011000;
                        default: glyph_row = 8'b00000000;
                    endcase
                end

                "U": begin
                    case (row)
                        3'd0: glyph_row = 8'b01000010;
                        3'd1: glyph_row = 8'b01000010;
                        3'd2: glyph_row = 8'b01000010;
                        3'd3: glyph_row = 8'b01000010;
                        3'd4: glyph_row = 8'b01000010;
                        3'd5: glyph_row = 8'b01000010;
                        3'd6: glyph_row = 8'b00111100;
                        default: glyph_row = 8'b00000000;
                    endcase
                end

                "Z": begin
                    case (row)
                        3'd0: glyph_row = 8'b01111110;
                        3'd1: glyph_row = 8'b00000100;
                        3'd2: glyph_row = 8'b00001000;
                        3'd3: glyph_row = 8'b00010000;
                        3'd4: glyph_row = 8'b00100000;
                        3'd5: glyph_row = 8'b01000000;
                        3'd6: glyph_row = 8'b01111110;
                        default: glyph_row = 8'b00000000;
                    endcase
                end

                " ": glyph_row = 8'b00000000;

                default: glyph_row = 8'b00000000;
            endcase
        end
    endfunction

    wire [7:0] cur_char  = msg_char(auth_mode, char_index);
    wire [7:0] cur_glyph = glyph_row(cur_char, char_row);

    wire text_pixel =
        in_text_box &&
        (char_index < (auth_mode ? AUTH_LEN : NOAUTH_LEN)) &&
        cur_glyph[7 - char_col];

    // ------------------------------------------------------------------------
    // Pass timing straight through; modify only pixel data
    // ------------------------------------------------------------------------
    assign m_vid_pVDE   = s_vid_pVDE;
    assign m_vid_pHSync = s_vid_pHSync;
    assign m_vid_pVSync = s_vid_pVSync;

    assign m_vid_pData =
        (!s_vid_pVDE) ? C_BLACK :
        (in_banner)   ? (in_border ? C_WHITE :
                         text_pixel ? C_WHITE :
                         banner_fill)
                      : (auth_mode ? s_vid_pData : C_BLACK);

endmodule