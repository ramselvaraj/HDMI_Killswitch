module colorbars (
    input  wire        pix_clk,
    input  wire        rst,
    input  wire [10:0] hcount,
    input  wire [9:0]  vcount,
    input  wire        de,
    output reg  [7:0]  red,
    output reg  [7:0]  green,
    output reg  [7:0]  blue
);

    wire [2:0] bar = hcount / 80; // 640 / 8 = 80 pixels per bar

    always @(posedge pix_clk) begin
        if (rst) begin
            red   <= 8'h00;
            green <= 8'h00;
            blue  <= 8'h00;
        end else if (!de) begin
            red   <= 8'h00;
            green <= 8'h00;
            blue  <= 8'h00;
        end else begin
            case (bar)
                3'd0: begin red <= 8'hFF; green <= 8'hFF; blue <= 8'hFF; end // white
                3'd1: begin red <= 8'hFF; green <= 8'hFF; blue <= 8'h00; end // yellow
                3'd2: begin red <= 8'h00; green <= 8'hFF; blue <= 8'hFF; end // cyan
                3'd3: begin red <= 8'h00; green <= 8'hFF; blue <= 8'h00; end // green
                3'd4: begin red <= 8'hFF; green <= 8'h00; blue <= 8'hFF; end // magenta
                3'd5: begin red <= 8'hFF; green <= 8'h00; blue <= 8'h00; end // red
                3'd6: begin red <= 8'h00; green <= 8'h00; blue <= 8'hFF; end // blue
                default: begin red <= 8'h00; green <= 8'h00; blue <= 8'h00; end // black
            endcase
        end
    end

endmodule